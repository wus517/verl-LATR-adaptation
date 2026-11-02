# LATR 旧代码实现与阅读记录

更新时间：2026-11-02。

范围：本地 `latr/` 的 LATR 核心实现，以及与迁移相关的 verl 接口。本文记录静态阅读结果，未进行 GPU 训练或端到端复现。

配套文档：[算法细节](latr_algorithm.md)、[verl 0.9.0 适配计划](latr_verl_0_9_adaptation.md)。

## 1. 仓库结构与来源

```text
项目根目录/
├── LATR.pdf             论文
├── latr/                基于 verl 0.5.0 的旧实现
│   ├── model/           LATR 核心生成、过滤、树状态
│   ├── tools/           配置转换、张量工具等
│   ├── data/            数据处理与 Countdown 奖励
│   ├── scripts/         GRPO/DAPO、kt/vllm、Math/Countdown 脚本
│   └── verl/            修改后的旧框架
└── verl-0.9.0/           目标框架
```

旧 [README](../latr/readme.md) 明确：`kt` 表示 lookahead tree rollout，即本文要迁移的 LATR；`vllm` 是标准随机采样；`eptree` 是不同的对照策略。

旧 [requirements.txt](../latr/requirements.txt) 指定 `verl==0.5.0`、`vllm==0.10.0`、`transformers==4.53.2`。这些是声明的依赖，不代表当前环境已安装。

另有现成讲解 [LATR_LOOKAHEAD_EXPLAINED.md](../latr/LATR_LOOKAHEAD_EXPLAINED.md)，可作补充；本组文档为后续迁移保存跨版本上下文，涉及行为以源码核对为准。

## 2. 核心文件索引

| 文件 | 关键对象/职责 |
| --- | --- |
| [scripts/grpo-kt-countdown.sh](../latr/scripts/grpo-kt-countdown.sh) | 较适合先理解编辑距离版本的启动配置 |
| [scripts/grpo-kt-math.sh](../latr/scripts/grpo-kt-math.sh) | Math 的不同阈值、距离指标与调度 |
| [verl/trainer/main_ppo.py](../latr/verl/trainer/main_ppo.py) | 训练入口、Countdown reward manager 注入 |
| [verl/trainer/ppo/ray_trainer.py](../latr/verl/trainer/ppo/ray_trainer.py) | prompt 复制、rollout、奖励与策略更新；另有事后多样性下采样逻辑 |
| [verl/workers/fsdp_workers.py](../latr/verl/workers/fsdp_workers.py) | 按 `rollout.name=kt` 构造 `KTRollout` |
| [verl/workers/rollout/kt_rollout.py](../latr/verl/workers/rollout/kt_rollout.py) | verl 到 LATR 的桥接、混合引擎 |
| [model/generation.py](../latr/model/generation.py) | `generate`、`_sample_new_sequence_with_rollout` |
| [model/utils.py](../latr/model/utils.py) | 配置、`BranchParamScheduler`、`MixedCache`、`BranchInfo` |
| [model/sample_filter.py](../latr/model/sample_filter.py) | `ProbFilter`、`RolloutFilter`、过滤器组合 |
| [model/metrics.py](../latr/model/metrics.py) | 后缀匹配与 ROUGE-L |
| [data/reward.py](../latr/data/reward.py) | Countdown 格式/正确性评分 |
| [tools/utils.py](../latr/tools/utils.py) | 配置转换、`get_repeat_interleave` 等 |

## 3. 旧版调用链

```text
scripts/*-kt-*.sh
  -> python -m verl.trainer.main_ppo
  -> RayPPOTrainer.fit
     -> 为生成重复 prompt，传递 global_steps
     -> actor_rollout_wg.generate_sequences
        -> fsdp_workers 中选择 KTRollout
        -> KTRollout.generate_sequences
           -> 设置 scheduler 的当前训练步
           -> 按 rollout micro batch 切分
           -> _generate_minibatch
              -> model.generation.generate
                 -> HF 逐 token 前向
                 -> 分叉 / 前瞻检查 / 递归剪枝
                 -> 整理每个 prompt 的轨迹并补齐
              -> 必要时 vLLM 完成剩余续写
              -> 包装 DataProto
     -> reward / old log prob / ref log prob / advantage / update
```

`fsdp_workers.py` 将 `self.actor_module_fsdp` 直接传给 `KTRollout`，`kt` 分支使用基础 sharding manager。旧集成与训练模块对象有直接耦合。

## 4. KTRollout 的行为

### 4.1 初始化

`KTRollout(module, path, config)`：

- 要求 `calculate_log_probs=False`；rollout 结果中的模型概率由 actor 后续重算。
- 从 `config.kt` 建立 `KeyTokenGenConfig`，设置 `sync_gpus=True`。
- 加载 tokenizer；缺少 pad token 时使用 EOS。
- 默认开启 `kt_mixed_engine`，创建进程内 `vllm.LLM`，支持 sleep/wake。
- 持有 `KtModules`，其中包括 scheduler 和可选的学习型差异判断器。

### 4.2 每次生成

- 从 `prompts.meta_info['global_steps']` 设置混合比例。
- 训练时按 `rollout.micro_batch_size` 切分；使用 vLLM 验证时不做该切分。
- 每次 `_generate_minibatch` 从 rollout 配置填写 `n`、temperature、top-k、top-p、response length。
- 验证时设 `fallback=True`，使用验证采样参数，绕过树形生成。
- 返回后恢复训练模式并清理缓存；vLLM 重新 sleep。

### 4.3 HF 到 vLLM 的交接

HF 可能只返回部分回答。此时：

1. 唤醒 vLLM，必要时把当前训练模型权重加载到 vLLM。
2. 把 `prompt + 已生成回答前缀` 转成 vLLM 的 `prompt_token_ids`。
3. 为每个前缀调用 `n=1` 的普通随机续写。
4. 最多生成 `response_length - HF前缀长度` 个 token。
5. 将续写与 HF 前缀拼接，计算最终 mask 和 position IDs。

交接只传 token，不传 HF KV cache；vLLM 会对交接前缀重新 prefill。旧权重更新通过 vLLM 私有对象路径和 `load_weights` 完成，不能直接视为新版公共接口。

## 5. 数据布局与组识别

旧 trainer 在生成前做 `repeat_interleave(n)`。典型输入形状：

```text
input_ids: [B * k, prompt_length]
行顺序: p0,p0,...,p0, p1,p1,...,p1, ...
```

`generate` 用 `get_repeat_interleave` 依据 token 内容推断重复间隔，再缩回每个 prompt 一行。根据混合比例：

```text
kt_n_gen   = round(k * mix_ratio)
orig_n_gen = k - kt_n_gen
```

- `kt_ids`：每个 prompt 先保留一条根路径，动态增加/删除行。
- `orig_ids`：每个 prompt 预先复制 `orig_n_gen` 条，普通随机采样。
- 一次模型 forward 的行布局：`[全部普通采样行, 全部当前树路径行]`。
- 返回时按 prompt/root 重新整理：该组的 KT 路径、KT 补齐路径、普通采样路径。

混合比例为零或验证 fallback 且允许交接时，核心函数可直接返回原输入，由 vLLM 完成整个回答。

不能照搬的假设：新版 V1 是按 prompt 进入 worker 后才启动多个 session；重复 token 内容也不是可靠的 group ID。旧 micro batch 若切断重复组，还可能破坏间隔推断或输出行数约定。

## 6. 每步 forward 与两种概率

`generate` 首步输入完整 prompt；后续步输入最后一个 token，并传入 `past_key_values`、完整 attention mask、`use_cache=True` 和 `logits_to_keep=1`。

普通轨迹：经过 temperature、top-k、top-p 处理，再 multinomial 采样。

树路径同时计算：

| 概率 | 用途 |
| --- | --- |
| `kt_probs_raw = softmax(raw_logits)` | top 候选、概率阈值、累计模型 log probability |
| `kt_probs_processed` | 特定 `sample_nk` 模式下的随机 fallback |

默认 `sample_nk='none'`，树阶段原路径沿最高概率候选前进。`full` 在组已满且仅一个有效候选时采样；`always` 在仅一个有效候选时采样。不要把默认树阶段解释成每步都随机采样。

## 7. 候选筛选与容量分配

`_sample_new_sequence_with_rollout`：

1. 对所有路径取 raw probability 的 top `max_n_branch_per_token` 候选。
2. 已结束路径不创建分支；活动路径由 `ProbFilter` 等筛选额外候选。
3. 最高概率候选始终有效。
4. 对每个 root 单独分配容量，超过 `kt_n_gen` 的候选标记为 suppress。
5. 原路径接上主候选；其余有效候选创建新行，并复制父路径前缀、attention mask、KV cache。

三态标记为 `0=保留`、`1=过滤掉`、`2=容量不足而抑制`。

重要实现细节：容量排序键为 `candidate_column * 当前组行数 + row_index`，优先保证现有行的主候选，再按候选名次及行顺序分配额外名额。没有对不同父路径的候选概率做统一全局排序。

`max_n_branch_per_token=2` 包括主候选，因此通常每条路径每步至多增加一个子分支。

## 8. 树与缓存状态

`Branch` 保存：

```text
parent             当前父路径行索引，根为 None
root               稳定的 prompt/root 标识
birth_step         出生步数
children           子路径行索引集合
accumulated_logp   生成 token 的累计模型 log probability
suppressed_num     因容量不足被抑制的候选计数
```

`BranchInfo` 负责按 root 分组、创建分支、查找后代、递归删除和行索引压缩。根路径不能被剪掉。

`MixedCache` 保存普通行数量 `orig_n_samples`，在旧 `DynamicCache.key_cache/value_cache` 的 batch 维上操作：

- 分叉：复制 KT 父行，索引需加普通行偏移。
- 剪枝：组合“普通行全保留 + KT 行筛选”的 mask，同步压缩各层 cache。

attention mask、序列、停止长度、树关系、cache 必须保持行对应。只改 token 序列而漏改 cache 会造成路径串线。

## 9. 前瞻时钟与过滤

代码出生步 `birth_step=step`，其中循环 `step` 从 0 开始；调用过滤器时传 `current_step=step+1`。

`RolloutFilter` 对各窗口 `r` 查找：

```text
birth_step == current_step - r
```

比较长度为 `current_step - birth_step`，取父子两行各自最后这么多个 token。按这套计时，比较片段包含分叉 token。复现时应保留或显式修改这个约定，并用边界测试固定，不能凭自然语言直接加减一步。

过滤器支持：

| 指标 | 剪枝条件 |
| --- | --- |
| 归一化编辑距离 | `< edit_dist_thres` |
| 后缀匹配相似度 | `> suffix_match_thres` |
| ROUGE-L 相似度 | `> rouge_l_thres` |
| 学习型过滤器 | 可选扩展，默认主配置未启用 |

多种指标同时启用时，任意一种触发即可进入删除集合。随后 `BranchInfo.remove` 删除该分支和所有后代。

实际例外：父子任一比较片段含 EOS 时跳过该次检查；默认 `keep_math_symbols=True` 还会排除部分数学 token 对应的比较。

## 10. 提前退出、同步与补齐

代码不是“任意一个 prompt 达到 k 就立即退出”：

1. 每个组统计有效路径数；该组 KT 路径全结束时，将该组视为已满。
2. 整个当前 batch 判断：

   ```text
   threshold = return_nb_thres_init
               - return_nb_thres_decay * step / max_new_tokens
   sum(group_effective_width)
       >= round(kt_n_gen * number_of_roots * threshold)
   ```

3. 检查存活分支是否已越过最长前瞻等待窗口；源码条件为存在 `birth_step + max_window >= step` 时继续等待。
4. `sync_gpus=True` 时用 `all_reduce(MIN)` 同步是否可提前退出，要求各 rank 同时就绪。
5. `force_return_step` 单独控制循环硬上限，可能使年轻分支来不及完成全部检查。

结束后按首次 EOS 截断有效部分、填 pad，再按 root 整理。KT 路径不足时默认复制该 root 的第一条路径补齐；复制前缀后可由 vLLM 独立随机续写。

`force_return_step` 与 `max_new_tokens`、`return_on_full`、是否存在续写引擎必须配套配置。纯 HF 模式不能直接沿用混合引擎的提前返回设置。

## 11. 参数与脚本快照

| 参数 | Countdown 的 kt 脚本 | Math 的 kt 脚本 |
| --- | --- | --- |
| `n` | 8 | 8 |
| `response_length` | 1024 | 8192 |
| `max_n_branch_per_token` | 2 | 2 |
| `prob_filter_abs_thres` | 0.25 | 0.15 |
| `prob_filter_rel_thres` | 0.15 | 0.25 |
| 编辑距离阈值 | 0.4 | 关闭 |
| 后缀匹配阈值 | 关闭 | 0.2 |
| ROUGE-L 阈值 | 关闭 | 0.5 |
| `rollout_filter_steps` | `[20,30,50]` | `[20,30,50]` |
| `mix_ratio_schedule` | `{0:1,10:0.75,20:0.5,80:0.25,200:0}` | `{0:1,100:0.75,200:0.5,300:0.25}` |
| `return_nb_thres_decay` | 0.05 | 0.05 |
| `force_return_step` | 512 | 1024 |
| `enable_param_scheduler` | False | False |

`BranchParamScheduler.set_step` 是分段常数选择，不做线性或指数插值。另一个自动调阈值的开关 `enable_param_scheduler` 与混合比例调度不同；脚本关闭它不代表混合比例不变。

## 12. 奖励与外围修改

- Countdown 的 `CountdownMathRewardManager` 解码“有效 prompt + 有效 response”，提取 assistant 后最后一行的 `<answer>...</answer>`，核对数字使用与算式结果。
- 得分写到最后一个有效回答 token；返回 `acc=(score==1.0)`，可供组过滤使用。
- 不能把该解析函数直接改接到“仅解码 response”的新接口而忽略 assistant 标记假设。
- 旧 trainer 的 `gen_n != n` 分支调用 `DiverseFilter`，先多采样再筛完整回答，属于对照/消融功能，不是 LATR 必需流程。
- 旧仓库还修改了验证长度统计、运行环境、分布式超时、`DataProto.chunk` 等；迁移应逐项判断必要性，不能整体覆盖新版 framework。
- `model/diverge.py` 等学习型过滤扩展不是论文主配置必需项，可暂不迁移。

## 13. 已知问题与待验证项

### 13.1 静态确认：数学符号过滤导致索引列表错位

`RolloutFilter` 先建立三个对应列表：`model_batch`、`candidate_ids`、`candidate_lens`。数学符号规则只过滤 `model_batch`，后续仍 `zip(candidate_ids, candidate_lens, model_batch)`。

当中间某个比较样本被过滤掉、后面仍有比较样本时，会把后者的距离结果配到前者的分支 ID，且比较长度可能错误。迁移应将 ID、长度和样本作为同一条记录筛选，保留最小回归用例。

本项为源码数据对应关系可以直接推出的问题；没有在 GPU 完整训练中验证其影响范围。

### 13.2 行为边界：最终轨迹不一定全通过检查

EOS 跳过、数学符号例外、强制退出和复制补齐都会影响检查覆盖。应统计并如实描述，不能视为所有返回轨迹都验证了差异。

### 13.3 需要测试固定的旧接口假设

- `get_repeat_interleave` 用 token 相等推断组，无法替代显式 group ID；完全相同的不同 prompt 实例也可能混淆。
- micro batch 边界、组宽、输出行数与输入重复行数之间的关系需要校验；wrapper 构造 position IDs 时存在行数必须相等的隐含假设。
- `BranchInfo.remove` 更新 `stop_lens` 时的尾部清零切片，需覆盖 `orig_n_samples>0` 且多次增删路径的情况，验证是否残留停止状态。
- 关闭混合引擎时，提前退出和强制退出与“必须返回完整长度”的断言可能冲突。
- vLLM 续写前应明确区分已 EOS 路径，防止把 padding 或已结束回答作为待续写前缀继续生成。
- 同步 HF forward、micro batch 次数、all-reduce 与权重收集必须保证 rank 一致，不能仅靠本地 batch 逻辑假设安全。
- 学习型过滤、统计开关和非主线 `sample_nk` 模式没有完整运行验证，不应声明可直接支持。

这些条目是静态阅读发现的验证需求，除 13.1 明确的数据列表错位外，不统一宣称已复现 bug。

## 14. 与新版的直接冲突

| 旧实现依赖 | 本地 verl 0.9.0 情况 |
| --- | --- |
| `fsdp_workers.py` 构造 KTRollout | 目标目录中该文件已不存在，训练 worker 位于 `engine_workers.py` 等 |
| 直接拿训练 HF module forward | 新版 rollout 服务与训练 engine 生命周期分离 |
| `BaseRollout()` 无参初始化 | 新基类需要 config、model_config、device_mesh，并有异步抽象方法 |
| 进程内 vLLM 私有对象路径 | 新版采用 ServerAdapter、Replica、LLMServerClient 和 CheckpointEngine |
| 旧 `DynamicCache.key_cache/value_cache` | 新版要求 Transformers 5.x，必须按实际 cache API 改写并验证 |
| 重复行恢复 prompt 组 | V1 worker 按 prompt 启动多个 session，数据经 TransferQueue 流转 |

迁移设计与阶段验收见 [适配计划](latr_verl_0_9_adaptation.md)。
