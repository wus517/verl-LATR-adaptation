# LATR 混合推理实现

更新：2026-11-04。当前分支 `feat/latr-hybrid-cached`，基线 `290cdd1`。
旧版逐 token vLLM RPC 实现保存在 `archive/latr-vllm-token-rpc`，本分支没有恢复该后端或修改 vLLM 私有采样代码。

## 1. 已实现的执行顺序

1. `AgentLoopManagerTQ.prepare_latr_prompts` 使用原版 `SingleTurnAgentLoop` 的聊天模板与截断规则生成文本 token。
2. V1 sync trainer 等待 vLLM sleep；整批 rollout 进入树阶段。
3. 所有 FSDP2 actor rank 按相同顺序导出最新完整参数。选中的 rank 将 CPU 权重（普通参数 BF16，HF 要求的 recurrent FP32 参数保留精度）分桶送入同节点的独立 Ray HF actor；不经 trainer 搬运权重。默认只建一个树 actor，避免每个训练 rank 都额外持有一份模型。
4. 完成权重导出并卸载优化器后，各 rank 同步，树 actor 才将完整 HF 模型放到自己的设备。训练参数由 FSDP2 CPUOffloadPolicy 管理。
5. 每个树 microbatch 的所有 prompt 各 prefill 一次；随后所有活跃分支合为一次单 token forward，`past_key_values` 持续保留。
6. 分叉通过重复父缓存行，剪枝通过删除缓存行；原生 `Cache.reorder_cache` 同时处理 KV、卷积状态、recurrent state。无回退、无完整前缀重复 forward、无逐 token Ray/vLLM 调用。
7. 完成当前 rollout 的全部树 microbatch 后，HF 模型回 CPU，树缓存释放，设备同步并清理 allocator cache。失败也执行该清理。
8. 同步同版本 actor 参数并唤醒原版 vLLM；每条未结束路径发送一次普通续写请求。这些请求并发提交，由 vLLM 自己合批；“一次性交给”指一次阶段切换，不是单个请求包含所有路径。
9. 原始 prompt 保持原长度；response 为“树前缀 + vLLM 后缀”，整个 response 的 mask 为 1。reward、TransferQueue、GRPO、actor/ref logprob 沿新版原有路径执行。

vLLM 和 HF **不交换缓存**，阶段交接只传 token；因此 vLLM 会对 prompt+树前缀重新 prefill 一次。

## 2. 算法行为

- 原始 logits 归一化得到候选概率，主路贪心。
- 默认绝对概率严格大于 0.25，与第一候选概率差严格小于 0.15；top-2 含主候选。
- 新分支按全组候选概率降序争夺容量，稳定 ID 维护父子关系。
- 20/30/50 token 比较窗口包含分叉 token；编辑距离归一化后严格小于 0.4 删除子分支及后代。
- EOS 保留真实终止位置，不补假 token；不足路径复制根前缀，再独立续写。
- `round(n * eta)` 条树路径，其余原版采样；eta 默认 `1 * 0.985**global_step`，也支持分段表。
- 树上限默认 512 token；达到容量且所有新分支经历最大窗口时可提前结束。
- 可选 suffix、ROUGE、数学符号保护默认关闭。
- validation 与显式 per-prompt greedy 不运行树。eta 为零或树上限为零时跳过 HF 导出与设备加载。

## 3. 代码位置

| 文件 | 职责 |
| --- | --- |
| `verl/experimental/latr/config.py` | 算法/执行参数与不支持组合的入口检查 |
| `core.py` | 无设备依赖的树状态、阈值、窗口、剪枝、预算 |
| `cached.py` | 批量 prefill/decode、缓存分叉重排、raw top-k |
| `tree_worker.py` | 独立 HF 模型、严格权重覆盖检查、设备加载/卸载 |
| `hybrid.py` | 整批阶段协调、同版本校验、原版 vLLM 续写 |
| `workers/engine_workers.py` | 集体导出、分桶传输、同节点树 actor |
| `workers/engine/fsdp/transformer_impl.py` | 可选 `raw_hf_keys` 导出，默认原路径不变 |
| `trainer/ppo/v1/trainer_sync.py` | 延迟 vLLM 唤醒、训练/验证生命周期 |
| `trainer/ppo/v1/agent_loop_tq.py` | 准备 prompt、消费整组树前缀、发布组状态 |

以上路径相对于 `verl-0.9.0/`。

## 4. 内存与性能边界

- 默认 `tree_worker_count=1`、`tree_micro_batch_size=1`。microbatch 是 prompt 数，每个 prompt 内分支仍批量 forward。可在容量允许时增加；没有偷偷减小 n 或比较窗口。
- 每个树 worker 必须在**单个设备**容纳完整 HF 模型、激活、全部分支缓存和重排行临时副本。9B BF16 权重约 18 GB（十进制），并非总内存需求。树阶段尚无 TP、paged cache 或共享页 COW。
- 每个树 worker 另持有 CPU 模型；分桶传输默认 128 MiB，上限为桶阈值加最大单张量，外加 Ray 序列化开销。跨 prompt 不永久保存树缓存。
- 一个 rollout 两次 actor 参数导出（树、vLLM）及 HF CPU/设备转移有固定开销；这比逐 token 外部 RPC 少得多，但尚无 NPU 实测加速比。
- Qwen3.5 在 NPU 树进程显式使用 HF 的 torch 卷积/DeltaNet fallback，避免误用可导入的 CUDA/FLA 内核。没有新增 NPU 融合 DeltaNet 内核；该部分可能成为性能瓶颈，不能承诺与专用推理引擎相同吞吐。
- NPU 原版 vLLM sleep 使用 level 1，是否完全释放设备权重/缓存要按训练机上的 vLLM-Ascend 验证。本实现沿用原版 sleep/wake，不修改其 allocator。
- 两阶段参数版本必须一致；框架/精度差异仍可能造成数值差异，同版本不表示逐位一致。

## 5. 首版支持范围

V1 sync、GRPO、FSDP2 CPUOffloadPolicy、优化器 CPU offload、BF16 dense 全量权重、单轮纯文本、原版 async vLLM、函数奖励。启用 ref 时也要求 FSDP2 offload policy。

主动拒绝 DAPO/group refill、跨多个本地更新的 parameter_sync_step、LoRA、QAT、MTP、教师/奖励模型、continuous-token、多模态、工具、多轮、PD、router replay、rollout correction/bypass、自定义 replay sampler/rollout skip。

没有提供 LATR 行为策略的精确概率密度；训练 logprob 必须由 actor 重算，不允许拿后缀采样概率冒充整条轨迹的行为概率。

## 6. 验证记录与后续验收

CPU 环境：本机 uv 缓存 Python 3.12、torch 2.14.0、Transformers 5.9.0，不安装/更新训练机依赖。新增用例和已有组任务等待用例见 `tests/experimental/latr/test_hybrid_on_cpu.py` 和 `tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py`。

本轮测试：19 项通过；Ruff、两个脚本的 shell 语法及 Hydra 参数组合、`git diff --check` 均通过。额外完成了真实小 Qwen3.5 checkpoint 的加载、两轮权重覆盖/树生成，以及 trainer 延迟唤醒/版本流转检查。

已验证：真实随机小 Llama、带 linear/full attention 的小 Qwen3.5，在多 prompt、不同 prompt 长度、分叉、剪枝后每一步缓存 logits 与完整未填充前缀计算一致；还覆盖 EOS、严格阈值、递归剪枝、组预算、阶段顺序、整段 mask、续写长度、版本错误、失败卸载与权重导出失败后集合通信排空。

待 NPU 验收：

1. 确认 CANN/torch_npu/Transformers/vLLM-Ascend 版本及 NPU 型号/数量/单卡容量。
2. 用 Qwen3.5-9B 实际 checkpoint 跑少量 prompt，测 sleep 后空闲内存及树阶段峰值。
3. 连跑至少两次 GRPO update，核验树与 vLLM 都使用更新后的参数，检查 loss/reward/梯度与组大小。
4. 验证保存与恢复后版本号一致；运行普通 validation。
5. 对比旧 RPC 分支和普通 GRPO 的 tokens/s、整轮时间、权重传输时间、树阶段时间、设备峰值内存。任何加速/节省数字以实测为准。
