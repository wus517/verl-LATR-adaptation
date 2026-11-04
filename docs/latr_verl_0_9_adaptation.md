# LATR 适配 verl 0.9.0 的工作计划

更新时间：2026-11-02。

状态：首版逐 token vLLM 请求实现被用户因效率问题否决。2026-11-04 用户新增“实现不麻烦、效率较高”的权衡要求，最新建议为“带持久缓存的 HF/NPU 批量树前缀 + 原版 vLLM 批量续写”，见 `latr_engine_design.md` 第 8 节；仍是建议，未实施。此前原生引擎树方案保留作长期优化。目标为 NPU、Qwen3.5-9B，具体设备/软件版本待提供。

配套文档：[算法细节](latr_algorithm.md)、[旧代码实现](latr_code.md)。

## 1. 用户目标与本轮范围

目标：依据本地论文与旧代码，在本地 `verl-0.9.0/` 上实现 LATR。

用户已要求开始实施；继续保留算法、旧代码、适配计划三份 Markdown 作为恢复上下文。旧 `latr/` 仅供参考，所有实现写入 `verl-0.9.0/`。

截至本次文档更新：

- [x] 阅读论文算法、实验设置与相关附录。
- [x] 梳理旧 `kt` 调用链、树状态、过滤规则和混合引擎。
- [x] 记录论文与实际脚本差异，以及旧实现问题。
- [x] 查看新版默认入口、V1/TransferQueue、rollout 接口和权重同步入口。
- [x] 建立三份可持续更新的上下文文档。
- [x] 确定首版行为基线与服务后端 B（见第 14 节）。
- [x] 实现独立树核心、vLLM 候选扩展及 V1 组级桥接。
- [x] 运行 CPU 单元/模拟边界测试、Hydra 组合与脚本参数检查。
- [ ] 在目标 NPU 验证候选、续写、baseline 与至少两步 LATR 训练。
- [ ] 验证保存/恢复、多卡稳定性和性能。

Git 状态：用户已建立独立的根仓库，分支 `main`，初始提交 `290cdd1 chore: initialize LATR adaptation workspace`。本轮改动位于该根仓库工作区，未提交、未配置或推送远程；不沿用旧 latr 的 Git 历史。

## 2. 以本地目标源码为准的事实

### 2.1 默认 V1 trainer

入口 [main_ppo.py](../verl-0.9.0/verl/trainer/main_ppo.py) 根据 `trainer.use_v1` 路由。本地 [ppo_trainer.yaml](../verl-0.9.0/verl/trainer/config/ppo_trainer.yaml) 默认：

```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: sync
```

`trainer_mode=sync` 表示同步训练节奏；它不等价于旧版进程内同步 rollout API。rollout 配置仍为 `mode: async`，使用服务接口。

旧 `ray_trainer.py`/V0 路径仍有源码，但不能把它误认为默认运行路径，也不能为省事直接恢复旧 `fsdp_workers.py`。

### 2.2 V1 的组与 TransferQueue 数据流

关键源码：[agent_loop_tq.py](../verl-0.9.0/verl/trainer/ppo/v1/agent_loop_tq.py)、[trainer_base.py](../verl-0.9.0/verl/trainer/ppo/v1/trainer_base.py)、[trainer_sync.py](../verl-0.9.0/verl/trainer/ppo/v1/trainer_sync.py)、[replay_buffer.py](../verl-0.9.0/verl/trainer/ppo/v1/replay_buffer.py)。

```text
TaskRunnerV1
  -> V1 trainer + AgentLoopManagerTQ
  -> 按 prompt 将任务交给 AgentLoopWorkerTQ
  -> _run_prompt 为该 prompt 启动 n 个独立 session
  -> 每个 session 调用 agent loop / LLMServerClient
  -> 结果后处理、reward、写入 TransferQueue
  -> prompt 下所有 session 结束后发布 finished/failure
  -> ReplayBuffer 收集完整组
  -> old/ref log prob、advantage、actor update
  -> CheckpointEngine 同步新权重
```

prompt 状态按 `uid` 管理，轨迹 key 形式为 `{uid}_{session_id}_{index}`。组结束状态必须在所有写入完成后发布，防止失败清理后仍有旧任务写入。

适配时需要让一棵 LATR 树协同产生该 prompt 的多条结果。直接把单条 `SingleTurnAgentLoop.run` 替换为独立树搜索、再启动 n 次，会改变预算和算法含义。

### 2.3 rollout 与权重接口

源码入口：

- [workers/engine_workers.py](../verl-0.9.0/verl/workers/engine_workers.py)：`ActorRolloutRefWorker`、rollout 构造与 checkpoint engine。
- [rollout/base.py](../verl-0.9.0/verl/workers/rollout/base.py)：新 `BaseRollout` 与 registry。
- [rollout/replica.py](../verl-0.9.0/verl/workers/rollout/replica.py)：Replica 与 `TokenOutput`。
- [rollout/llm_server.py](../verl-0.9.0/verl/workers/rollout/llm_server.py)：LLM 服务客户端与 manager。
- [vllm_async_server.py](../verl-0.9.0/verl/workers/rollout/vllm_rollout/vllm_async_server.py)：vLLM 服务生成与结果提取。
- [experimental/agent_loop/agent_loop.py](../verl-0.9.0/verl/experimental/agent_loop/agent_loop.py)：agent loop 基类、普通 manager 和后处理。

新 `BaseRollout` 要求 `resume(tags)`、`update_weights(...)`、`release()` 等异步接口，初始化需要配置与 device mesh；旧 KTRollout 不满足此契约。

配置支持自定义 `agent.agent_loop_manager_class`。它是候选接入点，但单独配置 manager 还不足以解决模型执行、组生成和权重同步。

### 2.4 依赖跨度

本地目标 [setup.py](../verl-0.9.0/setup.py) 声明：

- `transformers>=5.5.3,!=5.6.0,<5.11`。
- vLLM extra 为 `vllm>=0.18.0`。

旧代码声明 Transformers 4.53.2、vLLM 0.10.0。生成参数、DynamicCache、模型 forward 与私有 vLLM 对象访问都要重新验证。这些是源码依赖约束，尚未核查实际训练机器的安装版本。

## 3. 建议的最小目标

首版实施范围已采用以下约束：

- 先单轮文本、单个明确模型家族、GRPO，再扩展 Math 与 DAPO。
- 先适配新版默认 V1 的同步训练模式，rollout 使用新版服务生命周期。
- 先支持可审查的单 prompt 组生成，再考虑跨 prompt 合批和吞吐优化。
- 保持原有普通采样可运行，并让 LATR 可通过配置开关启用。
- 树内优先实现 raw-probability 分叉、编辑距离、多窗口、递归剪枝、组预算、普通采样混入和标准续写。
- 学习型差异模型、EPTree、RDS、随机分叉/剪枝消融、完全异步训练、工具调用和多模态暂不列入首版要求。

仅做一次生成不等于完成用户目标；最终仍需接通新版完整训练并验证权重更新。

## 4. 先固定“复现什么”

需要在编码前把配置基线写入实验配置和本文决策记录：

| 事项 | 候选 | 影响 |
| --- | --- | --- |
| 行为基线 | 论文参数 / 旧 Countdown 脚本 / 旧 Math 脚本 | 阈值、距离、退火不同 |
| 比例调度 | 指数 / 分段常数 | 各训练步 KT 路径数量不同 |
| 容量优先级 | 论文候选概率 / 旧代码名次与行顺序 | 分叉预算竞争结果不同 |
| 数学符号规则 | 关闭 / 保留但修正索引 | 剪枝覆盖与旧实现兼容性不同 |
| 硬性退出 | 无 / 可配 force-return | 前瞻完成程度、吞吐不同 |
| 首个运行环境 | 已知 NPU、Qwen3.5-9B；型号/卡数/显存未知 | 决定 TP/DP、batch 和可运行性 |

推荐先用 Countdown 的编辑距离规则验证核心，再提供独立命名的论文参数与旧脚本配置。不能把修改后的组合含糊命名为“完全复现原始 LATR”。

## 5. 模块划分建议

```text
LATR 配置与调度
        |
按 prompt 协调的组生成器
        |
树状态 + 分叉决策 + 前瞻过滤
        |
可替换的模型执行后端
        |
完整轨迹组
        |
新版 AgentLoopOutput / TransferQueue 适配
        |
新版 reward、ReplayBuffer、GRPO/DAPO update
```

建议把算法状态与框架桥接分开：

- 树模块：稳定 branch ID、parent ID、group ID、出生步、EOS、预算。
- 过滤模块：概率条件、距离条件、比较记录；避免三个平行列表错位。
- 后端模块：获得下一 token 分布、延伸路径、可选缓存操作、完成随机续写。
- 框架模块：prompt/session 映射、reward 后处理、TransferQueue 状态、权重版本。

建议使用明确的包命名空间，避免照搬顶层 `model`、`tools` 造成导入冲突。实际实现位于 `verl/experimental/latr/`，详见第 14 节。

## 6. 模型执行后端候选

| 方案 | 优点 | 实际约束 | 当前结论 |
| --- | --- | --- | --- |
| A. 独立 HF LATR 服务/worker，必要时接新版 vLLM 续写 | 容易保留完整 logits 和逐 token 树行为 | 需要独立模型显存、权重同步、cache 适配；不能随意取训练 engine 内部 module | 可作为正确性优先的候选，未确认资源可行性 |
| B. 扩展新版推理服务以支持树形操作 | 复用服务与推理引擎；有优化空间 | raw top 候选 + 显式前缀续写 + 组级协调；逐 token 请求较慢 | 首版已选并实现；真实设备能力与性能待验证 |
| C. 暂以旧 V0 trainer 作为桥接 | 部分历史结构更接近旧实现 | 仍然使用新 rollout 接口，旧 FSDP worker 不存在；容易形成长期维护分支 | 不作为默认推荐路径 |

### 6.1 方案 B 不能假定现有 logprobs 已足够

当前 `TokenOutput.log_probs` 仅暴露已选 token 的概率列表。vLLM 服务还把传入的 `logprobs` 当布尔值处理，转换为 `0` 或 `None`，返回时提取已选 token。

LATR 需要的是分叉点的多个 raw-probability 候选；默认 `logprobs_mode` 又是 `processed_logprobs`。因此不能简单传 `logprobs=2` 就声称已经获得 top-2 raw probabilities。

需要单独验证：

- 能否取得变换前的候选概率及其归一化语义。
- 能否在指定前缀上强制不同 token，再继续真实生成。
- 请求粒度是逐 token、分段还是服务内部循环；逐 token RPC 的成本是否可接受。
- 被保留、被剪枝、已结束的路径如何管理前缀缓存。
- prefix caching 与显式 cache clone 的实际差别，不能宣称二者完全等价。

## 7. 必须明确的数据契约

### 7.1 组输入

至少含：稳定 `uid/group_id`、tokenized prompt、attention/长度信息、训练步、验证标记、目标回答数、模型权重版本。

使用显式 group ID；不依赖 `get_repeat_interleave` 猜测 token 内容。保留两个不同样本恰好有相同 prompt 的情况。

### 7.2 组输出

每个输入 prompt 返回恰好目标数量的 session 结果，具备唯一且可恢复的 session ID。每条结果至少包含：

- 原始 prompt token IDs。
- 完整 response token IDs，包括 LATR 前缀和后续随机采样；不能把 LATR 前缀错放成不训练的 prompt。
- response/loss mask，正确处理 EOS、padding 和长度截断。
- 可选模型 log probabilities，必须说明来源与语义。
- rollout 类型、补齐标记、停止原因与必要统计。
- 与新版 staleness/control 兼容的模型权重版本字段。

对接现有后处理以生成 position IDs、reward 和 TransferQueue 字段。完整组成功写入后再标记 finished；失败时保证任务收敛，不发生迟到写入。

### 7.3 概率与 loss

- 首版建议复用 actor 对完整回答重算 old log probabilities，保持与旧实现接近的更新路径。
- 不能将强制选中 token 的条件概率写成 1 或 logprob 0 来代替原模型概率。
- 不启用需要尚未提供字段或语义不匹配的 bypass/rollout correction 配置。
- `π_model(token|prefix)` 不等于包含树选择/剪枝在内的完整 LATR 行为采样分布。即使重算 old log prob，也不能宣称消除了这类分布差异。
- 复现目标是实现论文/旧方法规定的训练行为；若引入新的分布校正，应作为显式算法变更单独评估。

## 8. 权重与资源生命周期

需要验证的约束：

1. 同一轮同步训练内，树阶段与随机续写阶段使用同一更新版本的策略权重。
2. 新 checkpoint 能同时更新所选 LATR 后端与续写后端，包含 resume 后的首次加载。
3. 训练和 rollout 在共享 GPU 时有可说明的 wake/sleep/offload 顺序；不能出现两份模型和训练 optimizer 同时超出显存。
4. 多 rank 的 collective 调用数量和顺序一致，不因不同树宽、EOS 或 micro batch 数而死锁。
5. 若保留 HF/vLLM 两阶段，明确 token 交接、重新 prefill 成本和上下文预算。
6. 最大上下文需覆盖 prompt 加完整 response，不能沿用旧 wrapper 仅设置 response length 的隐含假设。
7. 已结束路径不再发续写请求；剩余 token 数为零时直接完成，不依赖服务把零强制改成一。

在资源配置明确前，不承诺旧论文的 8×H200 性能或显存能直接迁移。

## 9. 分阶段工作与验收

### 阶段 0：确定基线与后端能力

- 核实目标 NPU 型号/显存、CANN、Python、PyTorch/torch_npu、Transformers、vLLM/vllm-ascend 版本。
- 选择论文或旧脚本行为基线，登记所有差异。
- 对候选后端做最小能力验证：raw 候选、固定前缀续写、权重更新、EOS/长度。
- 明确本地源码 `AGENTS.md` 等适用要求；实施前已阅读 `verl-0.9.0/AGENTS.md`，按现有入口与配置扩展。

验收：得到可运行的后端小样例与明确选型，不靠私有 API 假设直接进入训练集成。

### 阶段 1：提取与验证算法核心

- 将树状态、分叉、距离、调度抽成可独立验证的模块。
- 修正数学符号过滤索引错位；对其他旧边界先写最小复现再决定修复。
- 固定出生步与窗口计时、容量排序、补齐和退出行为。
- 适配目标 Transformers 的 cache API，或通过后端隔离 cache 实现。

验收：下节核心用例通过；每条路径的 token、父子关系和 cache 对应可追踪。

### 阶段 2：小模型真实组生成

- 单 prompt、少量回答、短长度，观察实际分叉和剪枝。
- 再验证多 prompt、相同 prompt 不同 UID、混合比例和验证 fallback。
- 检查两阶段拼接是否保留全部 LATR 前缀，是否重复/漏掉 token。
- 验证 tokenizer、左 padding、EOS、上下文上限与随机种子。

验收：每个 prompt 的输出数量、长度和 mask 正确；模型更新后生成使用新权重。

### 阶段 3：接通 V1 / TransferQueue

- 在 prompt 组级别协调 LATR，避免默认 n 个独立树任务。
- 接入现有 reward 与后处理，正确写入 session 结果及组状态。
- 验证 ReplayBuffer 按 UID 组成完整 group，不跨组混合。
- 先普通随机采样 baseline，再 LATR，完成少量完整训练步。
- 核对 old/ref log probabilities、优势、actor loss 和 checkpoint 同步。

验收：梯度更新完成且有限值正常；失败/取消能正确收敛；η=0 与验证能回到标准生成路径。

### 阶段 4：Countdown GRPO 端到端

- 使用小规模数据先 smoke test，再扩大 batch/长度。
- 核对奖励解析器与新版 prompt/response 接口。
- 比较普通 rollout 与 LATR 的轨迹差异、奖励分布、有效优势比例、耗时和显存。
- 验证保存、恢复及训练步恢复后的混合比例调度。

验收：训练和验证完整可运行，配置、命令、依赖和实际结果入文档。小规模成功不表述为论文指标复现。

### 阶段 5：DAPO、Math 与性能

- 使用新版组过滤路径，不复制旧 trainer 的整个训练循环。
- 逐项设置 clip、KL、loss aggregation、生成 batch 与 reward metric；方法名不能代替配置核对。
- 为论文参数和旧 Math 脚本建立分开的实验配置。
- 在正确性通过后做多 prompt 合批、缓存优化、多 GPU 与吞吐优化。

验收：组过滤可补足训练数据，不损坏组生成；记录完整训练指标与对照结果，再讨论论文效果复现。

## 10. 核心验证用例

以下为迁移验收清单；第 14 节说明已运行的 CPU 用例及其与真实设备验收的区别。

| 用例 | 必须验证的结果 |
| --- | --- |
| 绝对/相对阈值边界 | 严格 `>` 与 `<`，主候选保留 |
| raw 与 processed probability 不同 | 分叉仍使用约定的 raw 分布 |
| 多 root 且容量不足 | 每组独立预算，排序规则确定 |
| 20/30/50 窗口 | 触发步准确，不漏检、不重复检 |
| 嵌套分支被剪 | 后代全部删除，其他组不受影响 |
| 数学符号跳过 | 比较样本、ID、长度始终对应 |
| 分叉和多次剪枝 | token、mask、cache、停止状态保持对应 |
| η=0、η=1、混合与调度边界 | KT/std 数量正确，恢复训练不重置比例 |
| EOS 与提前退出 | 已结束回答不续写，未满组按约定补齐 |
| 组不足复制 | 输出数量正确且有复制标记，不宣称唯一性 |
| 同文不同 UID | 两个组互相独立 |
| 两阶段拼接 | LATR 前缀和后缀完整进入 response/loss |
| 新权重同步 | 两阶段版本一致，更新后不继续用旧参数 |
| TQ 写入失败/取消 | 终态顺序正确，无迟到 session 写入 |
| 多 rank 不同树形 | collective 顺序一致且能完成 |

## 11. 建议记录的指标

- 各组实际 KT/std 数量、活动树宽与最终有效路径数。
- branching/pruning/suppression 的计数及明确分母。
- 各窗口检查次数、EOS/数学符号跳过次数、年轻分支强制返回数。
- 补齐路径数、树阶段结束长度与退出原因。
- rollout 回答重复率、距离统计、奖励组方差和非零优势比例。
- HF/服务端生成、RPC、prefill、距离计算、权重同步耗时。
- GPU 峰值显存、整体每步耗时、训练/验证正确率和回答长度。

旧核心虽然构造了一些统计，但 wrapper 未完整上报；且部分旧统计分母/累计方式需要复核。不能直接照搬字段名并假设其含义正确。

## 12. 决策与后续更新记录

| 日期 | 记录 | 状态 |
| --- | --- | --- |
| 2026-11-02 | 用户要求先阅读，再逐步适配 | 已执行阅读 |
| 2026-11-02 | 用户要求算法、代码、适配计划分别持久化 | 已建立本组文档 |
| 2026-11-02 | 新版默认 V1 + TQ；旧 KTRollout 不能原样复制 | 已静态确认 |
| 2026-11-02 | 旧数学符号过滤导致索引列表错位 | 新树核心以稳定 ID 成对比较，避免平行列表错位；CPU 已测试 |
| 2026-11-02 | 首版后端 B、论文 Countdown 阈值、NPU + Qwen3.5-9B | 已实现；设备规格待补充，真实训练待验证 |
| 2026-11-02 | 独立根 Git 仓库 | 用户已完成初始化，初始提交 290cdd1 |

后续每次实际决策或实现后，更新本表和阶段勾选；记录修改文件、验证命令、真实结果与未解决项。只有运行过的检查才能写成“通过”。

## 13. 下一轮恢复上下文的入口

1. 阅读 [算法细节](latr_algorithm.md)，再读 [旧代码实现](latr_code.md)。
2. 查看本文状态与决策表，区分建议和已确认选择。
3. 阅读第 14 节和 `examples/latr/README.md`，检查工作区与实际测试结果。
4. 优先阅读第 16 节与 `latr_engine_design.md`：保留逐 token 算法但在引擎内执行，落实持久共享缓存；不要沿历史方案扩大训练，也不要重复初始化 Git。


## 14. 2026-11-02 首版实施记录

### 已实现文件与调用链

```text
main_ppo -> validate_training_config
  -> AgentLoopWorkerTQ._run_prompt
     -> run_latr_group（每个 UID 一次，目标 n 条）
        -> LATRGenerator（分叉、窗口、剪枝、混合、补齐）
           -> VLLMTokenBackend
              -> LLMServerClient -> vLLMHttpServer.generate
                 -> raw top candidates / 显式前缀续写
        -> AgentLoopOutput -> 原有 _agent_loop_postprocess
  -> 原有 ReplayBuffer / actor-ref logprob / GRPO / CheckpointEngine
```

- `verl/experimental/latr/config.py`：参数、指数/分段调度和不兼容训练配置拒绝。
- `core.py`：无 torch 依赖的树生成器、稳定分支 ID、递归删除、EOS/长度、并发任务收敛。
- `backend.py`：共享 worker 并发上限，prompt+response prefix 请求；树与续写全部检查同一非空权重版本。
- `vllm_contract.py` + vLLM 服务 `generate`：内部候选数量独立字段，保留 top-k 数量而非布尔转换；要求 raw logprob，不对 top 候选重新归一化。
- `agent_loop.py` + V1 `_run_prompt`：组级接入，完整 response/mask；验证与显式 greedy fallback 保留原路径。失败/取消时等待子任务结束，再发布组终态。
- `workers/config/rollout.py`、rollout YAML、`utils/config.py`：配置接入，默认关闭。
- `examples/latr/`：数据格式转换、新 reward、通用脚本、Qwen3.5-9B NPU 脚本、后端 probe 和运行说明。
- `tests/experimental/latr/test_latr_on_cpu.py`：标准库测试；真实框架方法使用模拟外部服务执行，不需要 torch/Ray/vLLM。

### 已固定的行为与差异

- 论文 Countdown 阈值 `.25/.15`、编辑距离 `.4`、窗口 `[20,30,50]`。
- 保留旧实现的“窗口包含分叉 token”计时；容量竞争选择论文式全局备选概率排序，不复刻旧 rank/row 遍历顺序。
- 默认数学符号豁免关闭；可启用且没有旧平行列表错位。suffix/ROUGE-L 为可选项。
- `η=1*0.985**global_step`，恢复后根据训练步计算；每 prompt `round(n*η)` 条树路径，提供分段常数覆盖。
- 树硬上限 512 token，饱和比例 1.0、不衰减；可配置旧式饱和松弛。这是显式组合基线，不宣称完整重现所有旧脚本。
- 不做 HF cache clone；前缀由服务重新处理。Qwen3.5 初始关闭 prefix caching，可能很慢，未声称性能等同旧混合引擎。
- 训练完整回答；不提供树策略行为概率，不启用 rollout correction 或 bypass。
- 每 session `extra_fields.latr` 保存分叉/剪枝/补齐/窗口/退出统计，尚无 console 聚合。
- 首版支持 V1 同步 trainer、内置 vLLM async 服务、单轮文本 GRPO；未验证 Math/DAPO、多模态、工具、完全异步或自定义服务。

### 已运行的验证及实际边界

本机 macOS / Python 3.14，没有 torch、Ray、vLLM 和 NPU。核心测试使用标准库；Hydra/OmegaConf 与 Ruff 从本机已有 uv 缓存加载执行，没有变更训练环境依赖。验证结果：31 个测试全部通过（包含 Hydra 组合用例，无跳过）；Ruff check、脚本 bash -n、git diff --check 通过。标准测试命令见运行说明。

覆盖算法边界、20/30/50 时钟、子孙剪枝、容量排序、混合退火、EOS/补齐、前后缀拼接、UID 隔离、版本混用拒绝、生成失败和取消后的任务收敛、组发布与 mask、实际 vLLM 方法的候选数量传递、实际 shell 参数及 Hydra YAML 组合。真实模型 logits、NPU 内核、Ray/TQ 集群、训练梯度和 CheckpointEngine 更新没有在本机验证。

### 下一步设备验收

按 [运行说明](../verl-0.9.0/examples/latr/README.md)：数据转换 -> raw top-2/prefix probe -> 普通 GRPO baseline -> 两步短树 LATR -> 完整窗口与验证 -> 保存/恢复 -> 性能。设备 probe 通过也不等于训练通过。记录 NPU 型号/数量/显存及软件版本后，再调整 TP、batch 和 offload。

## 15. 用户否决逐 token 请求后的新方案（设计，尚未实现）

2026-11-02：用户明确认为逐 token 请求 vLLM 效率过低，要求新的方案。第 14 节和现有启动脚本描述的是被否决的旧执行后端，不能当作已接受的最终架构。

### 推荐：专用 LATR rollout worker + Transformers/NPU 持久混合缓存 + 批量 decode

保留 V1 组级输入/完整输出、reward/GRPO 和权重版本约束，将模型执行整体迁入长驻 rollout worker。一次调用提交一个 prompt microbatch 及各组目标 n，返回完整回答；token 循环全部在 worker 内进行。树拓扑规则可复用，但现有“每 branch 调一次 backend.candidates”的执行接口必须替换为批量 step 接口。

1. 每个 prompt microbatch 对独立根路径 prefill 一次，保存完整模型状态。
2. 在同一个推理副本内，将多 prompt 的活动树路径合批；每步输入各路径最新的一个 token，加持久缓存，完成一次批量 forward。
3. 在 NPU 上由 raw logits 的 top-k 与全词表 logsumexp 计算真实候选概率。只传小规模候选/决策数据，不逐步传完整 logits 或全前缀。
4. 接受分叉时，把已处理共同前缀的父状态复制到空闲子槽，父/子分别写入不同 pending token，下个 decode step 才消费。剪枝同步释放后代的缓存槽、mask、position 和树元数据。避免混淆“已写入 token”和“已进入缓存的 token”。
5. 缓存包括 attention K/V、Gated DeltaNet convolution state、recurrent state；它们必须一起 fork/select/release。初版允许在分叉/剪枝时做缓存行复制，不要求立即实现 paged copy-on-write；不可每 token 深拷贝整份缓存。
6. 第一阶段让普通采样、树阶段及树结束后的随机续写都在同一 worker、同一模型中完成，直接复用缓存，避免 HF/vLLM 双模型生命周期。后续如实测显示有收益，再加入“整棵树结束后仅交接一次到 vLLM”的可选混合方案；该交接会重新 prefill。
7. 对接新版 rollout replica/server 生命周期及 CheckpointEngine 权重接收，训练/推理按阶段 wake/offload；不直接访问 actor 的内部 FSDP module。多卡副本必须用统一的分支决策与 collective 顺序；首个后端实现优先明确副本内并行策略，不能假定 9B 单卡可容纳。

这会消除逐 token RPC 与反复全前缀 prefill，保留自回归模型本身每 token 的 decode forward。仍有缓存复制、动态树宽、host/device 决策同步与 NPU 算子效率成本；吞吐需实测，不能承诺等同 vLLM 或某个加速倍数。

### Qwen3.5 缓存接口证据与首要验收

本次静态检查发现本机 uv 缓存中的 Transformers 5.9.0：Qwen3.5 使用按层混合 DynamicCache，linear attention 层含 conv_states/recurrent_states，reorder_cache 通过 index_select 选择批次行。但本 checkout 的部分训练 patch 仍访问另一种 cache 布局，目标 NPU 软件版本未知。此发现仅证明有可适配的接口，不代表当前训练 patch 或实际设备模型可直接用于 cached decode。

需要独立 inference 模型路径和版本明确的 cache adapter，不盲目套用训练 forward patch 或旧 MixedCache。首个设备验收是：同一共同前缀 fork 出不同 token，分别缓存续写，与两个独立完整前缀 forward 的下一步 logits 比较；再覆盖两子分支互不污染、多次 fork/prune、EOS/position/padding、多 prompt 批处理和权重更新后清空旧缓存。之后接通至少两步完整训练并记录性能。

### 长期备选：在 vLLM/vllm-ascend 引擎内部实现原生树调度

需要 scheduler、模型 runner/sampler、KV block manager 以及 Qwen3.5 recurrent-state manager 的 fork/free 协作，主干共享 KV block，写入时 copy-on-write，循环状态独立复制；剪枝释放资源，树结束直接切普通采样。不是把外部循环搬到 HTTP server，也不是仅打开 prefix caching。

该方案潜在吞吐更高，原有权重同步链更易保留，但要绑定目标机器的 vLLM/vllm-ascend 版本并维护引擎扩展。仓库未包含这两个引擎的源码，当前不能声称已有可用的 fork API，因此不作为最快完成正确适配的首选。

## 16. 进一步明确资源目标后的推荐（取代第 15 节推荐次序）

用户进一步强调算法逐 token 推进是必要的，要求减少额外时间和显存。当前推荐正式采用“vLLM/vllm-ascend 引擎内 LATR”作为目标架构；第 15 节专用 HF worker 仅作正确性对照候选。理由是共享分页 KV、复用单套 rollout 权重和现有推理 kernels/TP，比复制整行历史缓存和维护独立模型更贴合最终资源目标。

完整设计、分叉时机、Qwen3.5 状态所有权、容量约束与实施验收见 [LATR 高效 rollout 执行方案](latr_engine_design.md)。这是设计决策记录，尚未完成引擎扩展或设备验证；不得将规划接口当成已有 API。
