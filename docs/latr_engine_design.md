# LATR 高效 rollout 执行方案

更新：2026-11-04。状态：架构设计，尚未实现。目标为 verl 0.9.0、Ascend NPU、Qwen3.5-9B。用户新增“实现不麻烦、效率较高”的权衡要求，最新推荐见第 8 节；前七节保留引擎原生树方案作为长期优化设计。

## 1. 目标与选型

用户要求实现 LATR 本身的逐 token 分叉/前瞻/剪枝逻辑，同时降低时间和显存开销。逐 token 模型 decode 是必要工作；逐 token 外部请求、全前缀重复 prefill、整份历史 cache 复制不是算法要求。

正式目标选择：在 vLLM/vllm-ascend 引擎内增加 LATR 组调度与分支状态管理，复用其模型、批量推理、TP、缓存池和 verl 权重同步。专用 HF 解码器仅适合作为结果对照，不再是追求低时间/显存的主方案。第一个逐 token 服务请求后端保留历史记录，不再扩大训练。

这需要与确定版本的 vLLM/vllm-ascend 联合适配，不是仅修改 verl 的外层 agent loop。当前工作区没有这两者的引擎源码，不能假设现成公开 API 已支持所有操作，也不能承诺只用 worker_extension_cls 就能实现。

## 2. 调用与执行边界

```text
verl V1 / LATR group adapter
  -> 提交 prompt、UID、n、η、算法参数、采样参数
  -> vLLM engine（一次请求持有整个组）
     -> admission + prompt prefill
     -> 每步批量 forward / raw candidate extraction
     -> LATR group controller：fork / window compare / recursive prune
     -> cache-state manager：share / copy-on-write / state copy / free
     -> 原地切换普通随机续写
  <- n 条完整回答 + mask 所需信息 + 统计 + 权重版本
  -> 原有 reward / TransferQueue / GRPO / actor update
```

外层不等待每个 token，也不搬运全词表 logits。引擎内 token 循环与普通自回归生成一样持续运行。多个 prompt 组共享连续批处理；组内按逻辑 token 步维护窗口，受调度影响的分支不能简单用别的路径“最后 r 个 token”代替同一窗口。

## 3. 一个 decode 边界上的分叉

假设共同前缀 P 已消费并写入状态，模型产生预测下一个 token 的 logits：

1. 在设备上从 raw logits 取得 top-k 和全词表归一化量，得到原模型概率，保留算法严格阈值与容量排序。
2. 确认接受哪些分支后再分配子状态；仅有一个低概率备选不应触发任何缓存复制。
3. 父/子共享 P 的只读 attention KV blocks；各自持有逻辑 block table 和长度。共享的未满尾块在写入前复制，或采用后端允许的安全独占尾块策略，不能让两个路径写同一存储。
4. 为子路径复制 Qwen3.5 的当前 convolution/recurrent state，安排在父路径下一次原地修改之前完成，保持正确 stream/event 顺序。
5. 父/子分别挂接待消费 token A/B；下个批量 decode step 才把 A/B 写入各自状态。

不能把 fork 解释成消费 A 后再复制父状态给 B，也不能通过将 token IDs 写到 Python 列表就声称完成状态分叉。TP 副本内每个 rank 复制本地状态分片，采用一致的分支表/映射/collective 顺序。

## 4. 显存管理

### Attention 层

共享真实共同祖先对应的完整 KV blocks，以引用计数跟踪；只为分歧后 token 增加 blocks。分叉不复制全历史 K/V，剪枝递减引用并回收没有其他引用的 blocks。共同 token 内容相同但模型状态/权重版本不同，不视为可共享前缀。

### Gated DeltaNet 层

convolution/recurrent state 随消费 token 被更新。分叉后必须独立，通常复制当前固定形状状态，不能永久共用可写张量。该状态可能很大，必须按真实模型配置、dtype、TP 分片和后端 padding 计算，不假定可忽略。只保留每条活动路径当前状态；LATR 无回溯，不为所有历史 token 保留 recurrent snapshots。

### 结束与剪枝

EOS 路径只保留回答 token 和必要组元数据，模型状态可释放；如果需要补齐短缺组，则直接复制已结束的回答，或对仍需续写的根路径执行正确的状态 fork。剪枝递归释放后代的 KV 引用、状态槽与调度条目，不能只删树节点。

总推理内存近似为：

```text
模型权重
+ 唯一 attention KV blocks（含分支独占尾块与碎片）
+ 活动路径数 × 每路径 hybrid state
+ batch workspace / graphs / logits 与采样缓冲
+ 调度和通信缓冲
```

同一个 prompt 的树路径上限仍为算法规定的 k_LATR，其余普通路径不使总轨迹预算超过 n。引擎按 KV 和 hybrid state 两种资源联合准入，控制同时进入引擎的 prompt 组数；资源不足时排队或实施明确、正确的抢占恢复，不通过减少 n、压低窗口或丢弃合法分支来隐式修改算法。单组峰值都无法容纳时必须明确报错/调整设备并行，不能靠排队解决。

## 5. 时间开销管理

- 每 prompt 根路径 prefill 一次，普通路径也尽可能从相同根状态分出；恢复/抢占导致的重算单独计量。
- 各组活动路径跨组合批，使用现有 NPU 推理 kernels；保留标准续写的连续批处理，避免另起 HF 模型或跨引擎交接。
- raw 候选从已有 logits/sampler 路径提取，仅传 O(活动路径数 × k) 候选元数据；不将 full logits 复制到 CPU。
- 概率分叉每步判断；距离只在指定出生年龄窗口计算，比较小规模 token 窗口。首版可以在 engine controller CPU 侧计算精确编辑距离，复用 sampler 的小规模结果通路；保留同步时间指标，不宣称零 host/device 同步。
- window 到期的剪枝结果应在相关路径下一次推进前生效，保持算法顺序。可以采用有限 batch shape 桶复用图，避免为每种树宽重新编译；具体 Ascend graph 支持需按版本验证，不能先承诺。
- 树阶段结束后仅切路径的采样模式，继续复用现有缓存；不再提交新 prompt 请求。

不能用“每 20 token 才判断一次分叉”或“只看生成好的若干答案再筛选”替代 LATR。也不默认用普通 packed tree-attention mask 实现 Qwen3.5，因为线性注意力 recurrent state 不具备普通 attention mask 的全部语义。

## 6. 改动范围与接口

概念接口（设计名，不代表已有 vLLM API）：

- `generate_group`：一次提交一个完整组，复用现有 batch admission。
- `fork_state(parent, children)`：跨 attention KV 和 hybrid state 的原子分支操作。
- `release_paths(ids)`：正确释放调度与缓存资源。
- `apply_latr_step(candidates)`：更新分支、窗口检查、剪枝，再继续调度。

实现至少涉及 engine request/group metadata、scheduler、runner/sampler、KV cache manager、Ascend hybrid-state slots/copy，以及输出组装。verl 侧保留 V1 组输入/完整 response 输出，复用原来的 weight update/sleep-wake；任何更新必须在旧组结束后清理旧版本缓存、prefix 索引和 recurrent state，再接受新版本组。

现有纯 Python 树规则可作为参考/测试 oracle，但逐 branch candidates API 必须重构，不能仅替换 backend 类就宣称完成。

## 7. 实施顺序与验收

1. 确定目标环境 vLLM/vllm-ascend/torch_npu/CANN 版本与 Qwen3.5 模型配置；取得匹配源码，检查实际 scheduler/cache 布局。
2. 最小两分支引擎实验：prefill P，一次 fork 后分别消费 A/B；与独立完整前缀计算比较下一步 logits/概率（按精度给合理容差），验证父子互不污染、无重复 prefill。
3. 连续 fork/prune、递归释放、尾块 COW、EOS、不同长度/padding、多 prompt、TP 映射和资源恢复测试。记录真实唯一 KV block 数和 hybrid state 槽数，检测泄漏。
4. 加入完整 LATR 组调度、精确窗口和混合采样，同现有算法 oracle 对比受控 logits 下的结果。
5. 接回 verl V1，至少两步训练验证更新后的生成，再测保存/恢复和验证 fallback。
6. 对比同一环境的普通 vLLM GRPO 与 LATR：总 step/rollout 时间、prefill token 数、decode token 数（包含被剪路径）、active batch、KV/状态峰值、fork copy 时间、窗口判断/同步时间。用 profile 决定下一步优化，不预报倍数。

LATR 的被剪路径也确实执行过前瞻计算，最终输出同样 n 条并不代表推理量等于普通采样；此部分是算法开销，必须如实统计。

## 8. 2026-11-04：以开发复杂度和效率折中为目标的推荐

推荐回到旧实现已经采用的混合路线：Transformers/NPU 使用持久缓存批量完成 LATR 树前缀，随后整批交接给原版 vLLM 完成普通续写。不修改 vLLM scheduler/KV block manager，不逐 token 发送服务请求。第 1 节引擎原生树路线保留为长期优化，不作为当前最简实现路径；本节仍是建议方案，未获用户最终选定或完成实现。

- 外层一次提交 prompt microbatch；HF 模型 prefill 后，所有活动路径每步一次批量 forward。保留逐 token 概率分叉和原定窗口检查，不改成每若干 token 才分叉。
- 持久混合 cache 仅在实际 fork/prune 时做批次行复制/选择。Qwen3.5 的 KV、conv/recurrent state 必须一起处理；不用每步复制所有历史缓存，也不在首版实现分页 COW。因而显存仍高于理想共享前缀方案，不能宣称消除了所有前缀复制。
- 默认使用已约定的树阶段停止条件和硬预算，不为性能隐式缩短窗口；树完成后只传 prompt+树前缀 token 给 vLLM，并拼回完整回答。vLLM 对每条交接路径会重新 prefill 一次，不传 HF cache。普通采样路径与验证直接由 vLLM 处理。
- 按完整 rollout 批次安排“树阶段 -> 续写阶段”，microbatch 仅用于阶段内部。树前缀暂存在 CPU token 列表，完成所有树 microbatch 后统一交接，避免每个 microbatch 反复唤醒/卸载模型。
- HF 树阶段与 vLLM 续写阶段错峰驻留 NPU；清空树 cache 并卸载树模型后，再唤醒/加载 vLLM。必须核实目标 Ascend sleep/offload 是否实际释放内存，不能只调用 empty_cache；训练权重/优化器按阶段 offload。两模型副本可存在 CPU，但不要求同时占据 NPU；CPU 内存、PCIe/HCCS 搬运和权重加载时间计入成本。
- 同一个 rollout 中两阶段固定为同一 actor 权重版本，通过新版显式生命周期更新；不复制旧 vLLM 私有 load_weights 链路。若复用训练模型做树生成，需在整个树阶段一次性准备推理权重/参数驻留，不能让 FSDP 每个 token 全量聚合；具体接入需单独验证。
- 先测一个 prompt 组峰值能否容纳 9B 推理权重、分支 cache 和工作区，再定 microbatch。BF16 9B 参数本身约 18 GB（十进制，不含额外模块和缓冲）；未知 NPU 显存下不能保证单卡运行。若必须新增 HF TP 才能容纳，开发复杂度需重新评估，不包装成现成简单支持。

此路线复用旧混合架构和新版 vLLM 续写，主要新增工作聚焦在 Qwen3.5 cache adapter、批量树执行和 verl 生命周期。相对逐 token 请求可消除树阶段重复 prefill/RPC，相对 HF 全程生成可利用 vLLM 完成长续写；实际收益取决于树长度占比、缓存复制和阶段切换开销。它不保证比原生引擎树或所有场景的 HF 全程生成更快。
