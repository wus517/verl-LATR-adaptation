# LATR 适配 verl 0.9.0 的工作计划

更新时间：2026-11-02。

状态：调研与计划阶段。此文件记录候选设计与验收要求，不代表已选择最终后端，也不代表功能已实现或已测试。

配套文档：[算法细节](latr_algorithm.md)、[旧代码实现](latr_code.md)。

## 1. 用户目标与本轮范围

目标：依据本地论文与旧代码，在本地 `verl-0.9.0/` 上实现 LATR。

当前用户要求按步骤推进：先理解论文和代码，再做适配。本轮补充要求是将理解和计划持久化到 Markdown。

截至本次文档更新：

- [x] 阅读论文算法、实验设置与相关附录。
- [x] 梳理旧 `kt` 调用链、树状态、过滤规则和混合引擎。
- [x] 记录论文与实际脚本差异，以及旧实现问题。
- [x] 查看新版默认入口、V1/TransferQueue、rollout 接口和权重同步入口。
- [x] 建立三份可持续更新的上下文文档。
- [ ] 确定迁移行为基线与首个实现后端。
- [ ] 修改算法实现或新版训练代码。
- [ ] 运行单元测试、GPU 生成、训练或性能复现。

用户已要求建立独立的新 Git 仓库，`latr/` 仅作参考，不沿用它的仓库历史。检查发现 `latr/` 和 `verl-0.9.0/` 各自含 `.git`，且工作区均无未提交修改。

拟采用项目根目录统一管理：新分支 `main`；将两个旧 `.git` 备份到新仓库 `.git/legacy-repositories/`，源码按普通文件跟踪，不使用 gitlink；论文和本组文档一并纳入初始提交，不配置远程仓库。

当前状态：已新增根目录 `.gitignore`；`git init -b main` 因沙箱禁止写入 `.git` 失败，随后申请的提权执行又因自动审批服务返回 `404 model not found` 而未执行。尚未初始化根仓库、移动旧元数据或创建初始提交。恢复时先检查实际状态，再继续，不假定上述拟定步骤已完成。

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

以下是实施建议，尚未由用户选择为最终方案：

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
| 首个运行环境 | 模型、GPU 数量/显存、TP/DP | 是否能使用完整 HF 模型及批大小 |

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

建议使用明确的包命名空间，避免照搬顶层 `model`、`tools` 造成导入冲突。最终目录名和扩展方式待实现设计确定，不在当前阶段创建占位运行代码。

## 6. 模型执行后端候选

| 方案 | 优点 | 实际约束 | 当前结论 |
| --- | --- | --- | --- |
| A. 独立 HF LATR 服务/worker，必要时接新版 vLLM 续写 | 容易保留完整 logits 和逐 token 树行为 | 需要独立模型显存、权重同步、cache 适配；不能随意取训练 engine 内部 module | 可作为正确性优先的候选，未确认资源可行性 |
| B. 扩展新版推理服务以支持树形操作 | 复用服务与推理引擎；有优化空间 | 需要 raw top 候选、强制分叉、前缀管理及组级协调；现有接口不足 | 必须先验证 API 能力与性能 |
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

- 核实目标训练机器的 CUDA、GPU 显存、Python、PyTorch、Transformers、vLLM 版本。
- 选择论文或旧脚本行为基线，登记所有差异。
- 对候选后端做最小能力验证：raw 候选、固定前缀续写、权重更新、EOS/长度。
- 明确本地源码 `AGENTS.md` 等适用要求；本次文档位于根目录 docs，未修改框架源码。

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

这些是算法迁移所需的行为测试，不为当前仅新增 Markdown 编写代码测试。

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
| 2026-11-02 | 数学符号过滤导致索引列表错位 | 已静态确认，未修复 |
| 2026-11-02 | 首版后端、配置基线、运行硬件 | 尚未确定 |
| 2026-11-02 | 用户要求建立独立根 Git 仓库，旧 latr 仓库仅参考 | 已确认；初始化被环境权限及审批服务故障阻塞，旧仓库未改动 |

后续每次实际决策或实现后，更新本表和阶段勾选；记录修改文件、验证命令、真实结果与未解决项。只有运行过的检查才能写成“通过”。

## 13. 下一轮恢复上下文的入口

1. 阅读 [算法细节](latr_algorithm.md)，再读 [旧代码实现](latr_code.md)。
2. 查看本文状态与决策表，区分建议和已确认选择。
3. 下一步优先完成阶段 0：行为基线与后端能力验证。
4. 未经新的具体实施安排，不将本文计划写成已实现功能；用户后续要求开始适配时，沿本计划推进并及时修订。
