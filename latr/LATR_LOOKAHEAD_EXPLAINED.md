# LATR 代码讲解：在生成过程中分叉、前瞻与剪枝

本文基于当前仓库提交 `2adb0ed`，沿代码执行路径解释论文 **Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards** 的 rollout 实现。重点是生成过程中的 lookahead：它在哪里执行、何时分叉、如何比较后续轨迹、怎样删除分支，以及何时切换到 vLLM。

说明以仓库提供的 `kt` 训练脚本及默认混合引擎模式为主。示例中的 token 用字母或编号表示，不代表一个汉字、一个单词或一个完整推理步骤。本文根据源码分析；性能讨论不包含服务器 GPU 实测结果。已发现的实现问题单独列在后文，不将算法意图与代码实际行为混为一谈。

## 1. 实现位置与整体流程

本项目基于 **verl 0.5.0**，依赖 **vLLM 0.10.0**、**Transformers 4.53.2**。依据分别是 [requirements.txt](requirements.txt)、[verl/version/version](verl/version/version) 和 [readme.md](readme.md) 的 Training 说明。仓库内的 `verl/` 是包含作者修改的本地源码；版本号不能进一步证明它对应上游的哪个精确 commit。

论文的 LATR 对应配置：

```text
actor_rollout_ref.rollout.name=kt
```

调用链为：

```text
verl 的训练流程发起生成
    ↓
fsdp_workers.py 根据 rollout.name 选择 KTRollout
    ↓
KTRollout 将当前 actor 模型传入 model/generation.py::generate()
    ↓
Transformers/PyTorch 自定义生成循环：
    每轮模型前向 → 判断候选 → 创建分支 → 前瞻比较 → 剪枝
    ↓
满足退出条件，返回保留路径的 token 序列
    ↓
如果还没有达到回答长度上限，交给 vLLM 普通采样续写
    ↓
完整生成结果返回训练流程，之后计算奖励并更新策略
```

LATR 的核心是在真实生成过程中暂时保留多条可能的路径，观察它们接下来是否形成足够不同的轨迹，再决定是否继续保留。默认脚本采用 token 序列的编辑距离判断差异，并没有在这个循环中调用可验证奖励来判断哪条路径答得更正确。

前瞻生成的内容会作为最终回答的一部分保留下来。它不是先做一段独立模拟、丢弃模拟结果，然后退回分叉位置重新生成。

### 1.1 HF 阶段与 vLLM 阶段

本文的“HF 阶段”指使用 **Hugging Face Transformers 模型接口和 PyTorch** 执行的自定义生成阶段。代码在服务器本地 GPU 上直接调用模型，不是访问 Hugging Face 在线推理服务。

vLLM 是另一套推理执行引擎。在默认模式下，前半段动态分叉和剪枝由 HF 阶段完成，后半段普通续写由 vLLM 完成。两阶段使用当前策略对应的权重，切换前会进行权重同步。

本地下载的模型权重可以用于这两个阶段。例如将配置设为：

```text
actor_rollout_ref.model.path=/your/local/path/Qwen2.5-3B
```

目录包含模型配置、权重和 tokenizer 等必要文件，并且模型能被相应框架加载即可。

### 1.2 vLLM 是否为 lookahead 做了修改

从当前仓库的依赖和调用链看，没有发现 LATR 要求安装一个定制的 vLLM fork，也没有发现它通过修改 vLLM 的调度器、采样器或 attention kernel 来实现 lookahead。

仓库的 [verl/utils/vllm_utils.py](verl/utils/vllm_utils.py) 确实包含 MoE 权重加载适配，以及 `VLLMHijack` 对 LoRA 加载方法的运行时替换。因此不能笼统地说仓库“完全没有 vLLM patch”。这些适配服务于权重加载与同步，LATR 的分叉和前瞻代码则位于 `model/generation.py`，由 HF 模型执行。

## 2. 先建立数据模型：一行是一条路径

理解这份代码，最重要的是将“当前路径”与“曾经发生过的分叉位置”区分开。

| 术语 | 本文含义 | 代码中的对应 |
|---|---|---|
| prompt／根 | 一道输入题目，每个 prompt 独立维护一组树路径 | `root`、`group_by_root()` |
| 路径／分支 | 从 prompt 开始、不断增长的一条 token 序列 | `kt_ids` 的一行与一条 `Branch` 记录 |
| 分叉位置 | 某条路径选择不同下一 token、创建新路径的位置 | 新路径的 `birth_step` |
| 父分支 | 新路径复制前缀时所依据的那条路径 | `Branch.parent` |
| 分支容量 | 同一 prompt 最多同时保留的树路径数量 | `kt_n_gen` |
| 剪枝 | 删除一条已有路径及其后代 | `BranchInfo.remove()` |

一条原路径在分叉后仍然是一条完整的、会继续增长的序列。代码不会让它停在分叉位置，变成一个不再生成的内部节点。

假设原来只有路径 A：

```text
A：prompt + 已生成前缀
```

当前下一 token 有两个合格候选 `a`、`b`，代码让 A 接 `a`，另外复制前缀并接 `b`，新增 B：

```text
A：prompt + 已生成前缀 + a → 继续生成
B：prompt + 已生成前缀 + b → 继续生成
```

A 和 B 同时向前生成。以后检查 B 是否值得保留时，A 已经拥有与 B 同样进度的后续序列，可以直接拿来比较。

## 3. 逐 token 的生成循环

核心循环位于 [model/generation.py](model/generation.py) 的 `generate()`：

```python
wrapped_range = range(
    config.max_new_tokens if config.force_return_step == -1
    else config.force_return_step
)

for step in wrapped_range:
    outputs = model(...)
    # 计算概率、追加 token、创建分支、剪枝、判断是否退出
```

这里的 `step` 是本次生成中的 token 轮次，从 0 开始，不包含 prompt 长度，也不是训练迭代次数。

| 循环变量 | 本轮生成的回答 token | 本轮结束后的回答长度 |
|---|---|---|
| `step=0` | 第 1 个 | 1 |
| `step=1` | 第 2 个 | 2 |
| `step=19` | 第 20 个 | 20 |
| `step=29` | 第 30 个 | 30 |

每轮是 batch 中各条未结束路径分别追加一个 token。如果有 5 条路径，这轮可以产生 5 个 token，但每条路径的长度只增加 1。

第一轮模型处理完整 prompt。后续轮次利用 KV cache，通常只输入每行最后一个 token：

```python
outputs = model(
    input_ids=full_ids[:, -1:] if step and config.use_cache else full_ids,
    attention_mask=full_mask,
    past_key_values=cache.to_dynamic(),
    use_cache=True,
    return_dict=True,
    logits_to_keep=1,
    ...
)
```

训练循环中的 `global_steps` 是另一个计数，用来确定当前训练阶段的混合比例等参数。不要将它与上述 token 轮次混用。

## 4. 每一步寻找可创建的新分支

### 4.1 直接利用本轮模型输出的下一 token 概率

在树搜索启用的情况下，每轮都会对尚未结束的树路径检查候选。顺序是：

```text
当前前缀 → 一次模型前向 → 下一 token 的 logits／概率
                                      ↓
                               判断候选能否分叉
                                      ↓
                         将选中的 token 接到各条路径上
```

这里不需要为了“判断是否分叉”再额外调用一次模型。候选概率就是正常生成下一 token 时已经计算出来的结果。

代码先从原始 logits 得到概率，并选出 `max_n_branch_per_token` 个最高概率候选。这个参数限制的是每条路径在某一个位置考虑多少个候选，不是整棵树的路径数量。

### 4.2 概率筛选

[model/sample_filter.py](model/sample_filter.py) 中的 `ProbFilter` 对非第一候选应用两个条件。以仓库脚本常见配置为例：

```text
max_n_branch_per_token = 2
prob_filter_abs_thres = 0.25
prob_filter_rel_thres = 0.15
```

设第一候选概率为 `p1`，第二候选概率为 `p2`，第二候选需要满足：

```text
p2 > 0.25
p1 - p2 < 0.15
```

例如 `p1=0.40`、`p2=0.32`，两个条件都满足，可以考虑创建第二条路径。若 `p1=0.70`、`p2=0.10`，则不创建。

参数名中的 `rel` 在当前实现中对应“与第一候选的概率差”，并不是 `p2/p1` 这样的概率比值。代码中的比较也有严格边界：第二候选概率恰好等于 `0.25` 时不通过，概率差恰好等于 `0.15` 时也不通过。

这一概率筛选使用原始 logits 的 softmax 概率。普通采样另有 temperature、top-k、top-p 处理后的分布，两者在实现中分别保存。

### 4.3 原路径的延伸方式

通过概率和容量筛选后，默认 `sample_nk="none"` 的树路径使用第一候选延伸原行，其他有效候选建立新行。如果只有一个有效候选，默认也直接使用它。

代码还支持 `sample_nk="always"`、`"full"` 等回退采样选项。因此不应将所有配置下的树路径都描述为随机采样，也不应将原路径称为“奖励最好的路径”。默认情况下，这里主要依据当前 token 的候选概率行动。

## 5. 分支容量与 suppress

### 5.1 容量是同时保留的名额

设每个 prompt 要返回 `N` 条回答，当前训练阶段的树搜索比例是 `eta`，代码计算：

```text
K = round(N × eta)       # kt_n_gen，树路径的容量
O = N - K               # 普通采样路径的数量
```

例如 `N=8`、`eta=0.5`，则同一个 prompt 的树搜索部分最多同时保留 4 条路径，另外有 4 条普通采样路径。每个 prompt 的树最初只有一条根路径，再逐步分叉。

这 4 个名额包含原来的根路径，不是额外允许创建 4 条。名额不是一次性消耗的累计分叉预算：

```text
已有4条 → 剪掉1条 → 剩3条 → 后续位置又可以新增1条
```

已完成但尚未从数据结构删除的路径也占名额。代码并没有在路径输出 EOS 后自动将其移到独立的“完成区”并释放容量。

### 5.2 候选合格，不代表一定允许创建

概率筛选之后，代码再按每个 prompt 的容量限制候选总数。标记有三种：

```text
0 = keep：候选可以使用
1 = invalid：候选未通过筛选
2 = suppress：候选原本合格，但容量不足，阻止新增路径
```

`suppress` 阻止的是尚未建立的新路径；剪枝删除的则是已经生成了一段内容的旧路径，两者不是同一个动作。

容量不足时，代码先保留所有已有路径的第一候选，再按候选排名、行顺序分配剩余名额。它没有将所有路径的候选概率统一排序。

例如容量为 3，已有 A、B 两行：

| 行 | 第一候选 | 第二候选 |
|---|---|---|
| A，第 0 行 | `a1: 0.40` | `a2: 0.30` |
| B，第 1 行 | `b1: 0.50` | `b2: 0.40` |

假设四个候选都通过相应筛选。A 接 `a1`、B 接 `b1` 先占两个名额。剩余一个名额按行顺序给 A 的第二候选 `a2`，B 的第二候选 `b2` 被 suppress。

所以即使 `b2` 的概率高于 `a2`，仍可能不被创建。这份代码的容量分配不能理解为跨所有路径的全局概率择优。

## 6. 在普通 batch 中实现分叉

### 6.1 增加 token 序列的行

假设分叉前只有一行，形状是 `[1, 4]`：

```text
行0：[P1, P2, x1, x2]
```

本轮有两个有效候选 `a`、`b`。原行接 `a`，复制前缀的新行接 `b`：

```text
行0：[P1, P2, x1, x2, a]  ← 原路径A
行1：[P1, P2, x1, x2, b]  ← 新路径B
```

batch 形状变成 `[2, 5]`。下一轮模型同时处理两行，它们独立计算后续 token。

关键操作可简化为：

```python
# 从原行复制前缀，接上次候选，准备新行。
new_seq = torch.cat([current_ids[row : row + 1], candidate_token], dim=1)

# 原有各行接上各自的主候选。
current_ids = torch.cat([current_ids, main_tokens], dim=1)

# 将新增路径追加到行维度。
current_ids = torch.cat([current_ids, *new_branch_seqs], dim=0)
```

这是解释操作顺序的简化片段，实际变量名和周边逻辑以 `_sample_new_sequence_with_rollout()` 为准。

### 6.2 KV cache 与分支关系一起扩展

`MixedCache` 包装了 Transformers 的 `DynamicCache`。创建新行时，它会通过 `index_select()` 取出父行各层的 key/value，再通过 `torch.cat()` 追加到 cache 的 batch 维度。attention mask 同样复制扩展。

复制时 cache 对应共同前缀；下一轮分别输入 `a`、`b`，两条路径的 cache 才进一步分化。这样不用从 prompt 开始为新路径逐 token 重算全部历史。

与此同时，`BranchInfo.add_branch()` 记录新路径的父分支、所属 prompt、出生轮次和子分支集合。token 行、cache 行和分支记录需要保持对应关系。

此处通过普通 batch 行和 HF cache 管理多条路径，并没有使用 vLLM 的分页 KV cache 来执行树分叉，也没有使用自定义树形 attention kernel。

## 7. Lookahead：先继续生成，再比较轨迹

### 7.1 分支有各自独立的检查时钟

配置：

```text
rollout_filter_steps = [20, 30, 50]
```

表示针对每条新路径，在它从分叉位置开始累计生成了 20、30、50 个 token 时安排比较。这里包含分叉位置选择的那个 token。

代码在创建分支时保存：

```text
birth_step = 创建它的循环step
```

每轮生成后，过滤器收到：

```text
current_step = step + 1
```

于是该分支的比较长度为：

```text
length = current_step - birth_step
```

例如 B 在回答第 20 个 token 处创建，`birth_step=19`：

| 回答已生成的 token 数 | 当前循环 `step` | B 的累计比较长度 | B 的检查 |
|---|---|---|---|
| 20 | 19 | 1 | 暂无 |
| 30 | 29 | 11 | 暂无 |
| 39 | 38 | 20 | 第一次 |
| 49 | 48 | 30 | 第二次 |
| 69 | 68 | 50 | 第三次 |

所以 `[20,30,50]` 不是整条回答的绝对 token 位置，也不是每隔 20、30、50 步做一次全局检查。

### 7.2 比较分叉后的累积片段

过滤器找到需要检查的分支后，通过其 `parent` 字段确定比较对象：

```python
parent_idx = branch_info[idx].parent
length = current_step - branch_info[idx].birth_step

a = completion_ids[idx, -length:]
b = completion_ids[parent_idx, -length:]
```

`completion_ids` 已去掉 prompt，因此比较内容是回答片段。对于从回答第 20 个 token 分出的 B：

```text
前瞻长度20：比较A与B各自的第20～39个token
前瞻长度30：比较A与B各自的第20～49个token
前瞻长度50：比较A与B各自的第20～69个token
```

每次都从同一个分叉位置开始，窗口不断加长。它不是比较三个互不重叠的片段，也不是比较两条完整回答。

### 7.3 默认用归一化编辑距离剪枝

脚本常用：

```text
rollout_filter_edit_dist_thres = 0.4
```

代码计算 token 序列的编辑距离：

```python
ratio = edit_distance(a, b) / length
if ratio < self.edit_dist_thres:
    remove_set.add(idx)
```

例如前瞻长度 20、编辑距离 3，则比例为 `3/20=0.15`，新路径因过于相似被标记删除；若距离为 10，比例为 `0.5`，则这次保留。比例恰好等于 `0.4` 时也保留，因为代码只删除严格小于阈值的情况。

这里衡量的是 token 轨迹差异，不是数学正确性，也不是必然准确的语义距离。不同 token 的开头可能很快生成相似后续，LATR 用前瞻观察这种情况，而不是只凭分叉位置的两个 token 不同就一直保留两条路径。

如果一条路径正常参与全部三次检查，则任何一次失败都会被剪掉；通过 20-token 检查只是暂时保留，之后仍要接受 30、50-token 检查。EOS、数学符号规则、强制退出等例外见后文。

## 8. 嵌套分叉：父路径继续存在，后代单独计时

前瞻不是一个阻塞阶段。在 B 等待自己的 20-token 检查期间，B 仍可在新的位置创建后代。

假设 A 在回答第 20 个位置分出 B；B 在第 25 个位置遇到两个候选 `c`、`d`：

```text
回答位置：    20       21…24       25       26…………39

A：           a ─────────────────────────────────→ A继续

B：           b ───── 原有前缀 ─── c ─────────────→ B继续
                               └ d ─────────────→ 新增D
```

如果将沿 `c` 的后续称作 C，那么“C”只是对后续方向的称呼。在代码中，这一行仍然是 B，保留 B 的出生时间和父分支 A；只有接 `d` 的路径会建立新的 D 记录。

| 路径 | 分叉位置 | 父分支 | 第一次20-token检查时取的片段 |
|---|---|---|---|
| B，沿 `c` 继续 | 回答第20个token | A | A、B各自第20～39个token |
| D，接 `d` 新建 | 回答第25个token | B | B、D各自第25～44个token |

B 到累计长度 20 时，取的就是 B 当前这一行中沿 `c` 生成的片段，不会把 D 的内容混进去，也不会从 B、D 中选一个代表。D 用自己的出生时间独立接受检查。

这是代码里的分支数据模型与常见“内部节点分成两个叶子、内部节点不再生成”的树画法之间的重要区别。这里的父子关系记录的是路径复制的来源；父路径本身仍然是一条生成中的序列。

## 9. 剪枝后的行为与 EOS 例外

### 9.1 删除当前路径和整棵后代子树，不回退

当 B 与 A 的轨迹过于相似时，`BranchInfo.remove()` 会递归获取 B 的后代，然后删除相应 token 行、attention mask 行、KV cache 行和分支记录，并更新剩余父子索引。

```text
删除前：
A：已经向前生成到当前位置
└─ B：已经向前生成到当前位置
   └─ D：B的后代

删除B之后：
A：从当前位置继续生成
```

A 在比较发生之前就一直与 B 同步向前。删除 B 后，没有“回到 A 的分叉点”这个动作，也不在旧位置重新判断是否分叉。腾出的容量可用于剩余路径未来位置的新分叉。已删除路径消耗的计算量无法收回。

### 9.2 任意一侧含 EOS，就跳过该次轨迹比较

`RolloutFilter` 中有明确判断：

```python
if torch.any((a == self.eos_id) | (b == self.eos_id)):
    continue
```

也就是说，子分支或父分支的比较片段中任意一个含结束 token，都不参与这次差异比较。它没有额外要求这条已结束路径与父路径达到足够大的距离。

例如 B 的 20-token 检查通过，但在累计长度 25 时输出 EOS，则之后 30、50 的比较会因为片段含 EOS 而跳过。因此不能将这份实现描述成“所有保留回答都保证通过三次差异检查”。

HF 循环中，已完成路径不会再参与新分叉；为维持 batch 同步，代码可能继续追加 EOS。返回前会根据记录的停止位置将后面的内容替换为 padding。逻辑上的回答已经结束。

EOS 不会使一个后代免于祖先剪枝。例如 B 从 A 分出，D 从 B 分出，D 已输出 EOS；如果之后 B 与 A 的检查失败，则 B、D 一起删除，即使 D 自身已经形成完整回答。

## 10. 何时退出树搜索

退出树搜索不是因为代码证明“以后不会再出现有价值的分叉”，而是因为数量和前瞻等待达到设定条件，或者本阶段生成长度达到硬上限。退出时回答可以还没有结束。

### 10.1 路径数量达到阈值

对于当前 rank 的一个 minibatch，记：

```text
K：每个prompt的树路径容量，即kt_n_gen
B：该minibatch中树根／prompt组的数量，即bs_roots
L：回答长度配置，即max_new_tokens
```

每个 prompt 计算一个有效路径数 `E_i`：

```text
如果该prompt当前所有树路径均已结束：E_i = K
否则：                              E_i = 当前保留路径数
```

然后计算数量阈值：

```text
theta(step) = return_nb_thres_init
              - return_nb_thres_decay × step / L

数量条件：sum(E_i) >= round(K × B × theta(step))
```

默认 `return_nb_thres_init=0.98`；脚本常将 `return_nb_thres_decay` 设为 `0.05`。后者使数量要求随 token 生成进度适当放宽，它不直接修改每个 prompt 的容量 K。

单 prompt、`K=4`、不考虑阈值衰减时，`round(4×0.98)=4`，需要达到 4 条。更大的 batch 则可能允许部分空缺：例如 `K=8`、`B=8`、`theta=0.98`，所需总数是 `round(64×0.98)=63`，不是每个 prompt 都必须有 8 条。

这里使用 Python 的 `round()`；示例均避开 `.5` 边界。精确边界行为应以 Python 运算为准。

### 10.2 所有保留路径都越过最长前瞻等待窗口

数量足够后，代码还遍历全部现存树路径，要求：

```text
step > branch.birth_step + max(rollout_filter_steps)
```

例如最长窗口为 50，最新分支记录的 `birth_step=100`，则最早在 `step=151` 满足这个年龄判断。此轮结束后，回答张量长度是 152。严格说，这比它触发 50-token 比较的时刻还稍晚：该次比较在 `step=149`、`current_step=150` 时发生。

因此，“路径刚好填满”并不意味着立即退出。还要给最新分支足够的检查机会：

```text
路径达到4条
    ↓
继续生成，等待新路径经过前瞻窗口
    ↓
某次检查剪掉1条，只剩3条
    ↓
继续向前，在新位置分叉，恢复4条
    ↓
再次等待新路径的前瞻窗口
```

这里检查的是年龄，不会统计“每个分支是否实际计算并通过了三次距离”。如果某次比较由于 EOS 或数学符号规则被跳过，年龄仍然会增长。

### 10.3 所有 rank 同步满足

`KTRollout` 设置 `sync_gpus=True`。正常提前退出前，各 rank 使用 `all_reduce(MIN)` 汇总自己的准备标记，只有全部准备好才一起结束 HF 阶段，避免部分 rank 先切换到 vLLM。

代码变量名 `all_kt_complete` 容易造成误解：它表示上述数量和等待条件满足，并不表示所有回答都已生成到 EOS。

### 10.4 force_return_step 是独立硬上限

循环范围直接由下面的表达式确定：

```python
range(max_new_tokens if force_return_step == -1 else force_return_step)
```

所以 `force_return_step=512` 表示这个自定义生成循环最多执行 512 轮，即每行最多在本阶段追加 512 个回答 token。到上限即退出，不要求数量足够，也不要求所有分支完成前瞻。

例如某个分支直到第 505 个回答 token 才创建，那么第 512 个 token 后，它仍会被返回，尽管尚未达到第一个 20-token 检查点。达到上限本身不会触发一次额外剪枝。

当 `force_return_step=-1` 时，循环长度使用 `max_new_tokens`。当前实现并不是对两个长度取 `min()`，因此配置硬上限时应保证它与回答长度预算相容。仓库 Countdown 脚本使用回答上限 1024、本阶段上限 512；Math 的 `kt` 脚本将本阶段上限设为 1024。

正常提前退出由 `return_on_full` 控制；默认混合引擎初始化时会将它设为 `True`。此外，通用生成函数在非同步模式下支持所有序列达到停止条件后退出，但这不能直接套到设置了 `sync_gpus=True` 的默认 worker 上。

## 11. 返回路径与数量补齐

退出循环时，代码保留当时尚未被剪掉的路径，按 prompt 重新分组。默认 `fill_return_sequences=True`，如果某个 prompt 的树路径少于 K，就复制该组的第一条树路径补齐。

假设 `N=4`、`eta=1`，目标为 4 条树路径，但达到硬上限时只剩 A、B 两条：

```text
补齐前：
A = prompt + 前缀A
B = prompt + 前缀B

补齐后：
第1条 = prompt + 前缀A
第2条 = prompt + 前缀B
第3条 = prompt + 前缀A  ← 复制第一条
第4条 = prompt + 前缀A  ← 再复制一次
```

复制的是整段现有 token 序列，不会重新寻找分叉位置，也不是根据奖励挑选“最好”的路径。第一条是当前组中的第一行；在正常父子索引维护下，根路径不会被剪掉，它仍是该组的首条路径。

如果复制的是未完成前缀，后续分别调用 vLLM 普通采样时可能产生不同续写，但不保证不同。如果本来已是完整回答，则复制出来的回答内容就是重复的。

补齐用于满足每个 prompt 返回指定数量回答的接口要求，不能据此认为树搜索找到了 K 条具有差异的路径。这个动作也不会在返回时重新施加前瞻检查。

## 12. 混合引擎：从 HF 前缀交接到 vLLM

### 12.1 两个“混合”配置控制不同事情

| 配置 | 控制内容 |
|---|---|
| `rollout.kt_mixed_engine` | 是否创建并使用 vLLM 引擎，默认 `True` |
| `rollout.kt.mix_ratio_schedule` | 随训练阶段变化的树路径比例，即 eta |

即使 `eta=1`，全部回答都来自树路径，也仍可以在树搜索退出后由 vLLM 完成普通续写。

当 `0<K<N` 时，代码在 HF 阶段会同时维护树路径和普通采样路径：普通路径放在 `orig_ids`，树路径放在 `kt_ids`，它们合成 batch 一起前向。普通路径通过 `torch.multinomial()` 生成，不参与树分叉；树路径按前述逻辑处理。

因此，“有一部分普通采样回答”并不意味着这一部分在整个过程中都由 vLLM 执行。HF 阶段结束后，这两类保留前缀一起交给 vLLM 续写。

若当前比例使 `K=0`，默认 `return_on_full=True` 的生成函数会直接返回输入，不执行上述逐 token 循环，随后交给 vLLM 普通采样。验证时 worker 设置 `fallback=True`，默认混合模式也使用这一跳过树搜索的路径。

如果要研究纯 HF 执行，不能简单将 `kt_mixed_engine` 改为 `False` 就假定所有提前返回配置仍可使用：当前 worker 在无 vLLM 时会断言返回长度等于回答长度配置，需要配合关闭提前返回、调整循环上限等配置。本文的完整交接流程以默认混合模式为准。

### 12.2 权重同步、token 交接与普通续写

在 HF 阶段，vLLM 可以处于 sleep 状态。需要续写时，worker 唤醒它，并通过 `convert_weight_keys()` 和 vLLM 模型的 `load_weights()` 将当前 actor 权重同步过去。不能将 vLLM 理解为始终使用最初下载、未经更新的权重。

随后代码构造：

```python
vllm_inputs = [
    {"prompt_token_ids": _pre_process_inputs(self.pad_token_id, s)}
    for s in kt_seqs
]
```

`kt_seqs` 此时是整理后的生成结果，包含树路径、补齐项以及可能存在的普通采样路径。vLLM 接收的是：

```text
原始prompt的token + HF阶段保留下来的回答前缀token
```

剩余生成预算为：

```text
vLLM max_tokens = 回答长度配置 - HF阶段返回的回答张量长度
```

每个前缀以 `SamplingParams(n=1, ...)` 普通采样续写，返回的 token 再拼接到 HF 前缀之后。如果 HF 已经返回完整配置长度的张量，则 worker 直接返回结果，不进入 vLLM 续写。

**一旦进入这次 vLLM 续写，就不再创建 LATR 分支，也不再补做尚未完成的 20、30、50-token 差异检查。** 这尤其影响在 `force_return_step` 前不久创建的路径。

### 12.3 两阶段没有传递 KV cache

交接对象只有 token 序列，没有将 HF 的 `DynamicCache` 转成 vLLM cache。因此 vLLM 要根据收到的前缀，在自己的执行体系中进行 prefill，再开始生成新 token。

这里应区分两种复用：

```text
HF树内部：复制已有路径的HF cache，避免为每个新分支重算全部共同历史。
引擎切换：不传递HF cache，vLLM需要建立自己的cache。
```

代码开启 `enable_prefix_caching=True`，vLLM 可能复用自己已缓存的相同前缀块，减少请求间共同前缀的重复计算；这不等于直接复用 HF 阶段的 cache，也不能保证所有重复前缀都完全免除计算。

## 13. 一次包含嵌套分叉、剪枝和补齐的完整例子

为了让时序清晰，以下使用一个说明用的小配置，并假定讨论的比较没有被数学符号规则跳过，也没有受到后文索引问题的影响：

```text
一个prompt，N=4，eta=1，因此K=4
rollout_filter_steps=[20,30,50]
rollout_filter_edit_dist_thres=0.4
force_return_step=60
回答长度配置=100
```

1. 初始只有 A。回答第 20 个 token 处出现两个合格候选，A 接第一候选，新增 B 接第二候选。现在有 A、B 两行。
2. 第 25 个 token 处，B 再次遇到两个合格候选。B 接 `c` 继续，新建 D 接 `d`。现在有 A、B、D 三行，容量还剩一个名额。
3. 第 39 个 token 生成后，B 从出生位置累计了 20 个 token。取 A、B 各自第 20～39 个 token 比较。假设差异比例为 `0.5`，B 暂时保留；D 的计数仍独立进行。
4. 第 44 个 token 生成后，假设 D 恰好输出 EOS。D 与 B 的 20-token 比较因 D 片段包含 EOS 而跳过。D 的回答逻辑上结束，但仍是 B 的后代。
5. 第 49 个 token 生成后，B 到达 30-token 检查点。假设 B、A 都未输出 EOS，而差异比例下降为 `0.3`，则 B 被标记剪枝。删除动作同时移除 B 和已经结束的 D，只剩 A。这里并不矛盾：不同长度窗口的归一化距离可以不同。
6. 假设之后一直没有合格的新分叉，到第 60 个回答 token 后达到硬上限。代码保留 A，结束 HF 循环，不回退、不重新搜索旧位置。
7. 目标是 4 条树路径，当前只有 1 条，代码复制 A 三次，得到 4 个相同的 60-token 回答前缀。
8. 将这 4 个前缀交给 vLLM，各自最多续写 `100-60=40` 个 token。随机采样可能产生不同后续，但没有 LATR 的差异检查来保证它们不同。

这个例子说明：原路径与新路径是并行生成的；后代独立计时却依赖祖先存活；硬上限可以终止尚未填满容量的树搜索；补齐确保数量，但不保证前缀多样性。

## 14. `kt` 与 `eptree` 的区别

仓库中另有 `rollout.name=eptree`。它选择 `EptreeRollout` 和 [model/eptree.py](model/eptree.py)，不是 LATR 的另一种同等后端。README 明确将论文 LATR 对应到 `kt`。

| 对比 | `kt`，即本文的LATR | 当前仓库的`eptree` |
|---|---|---|
| 选择分叉位置 | 生成时逐token检查下一候选概率 | 先生成到EOS或长度上限，再从已有轨迹中选位置 |
| 扩展方法 | 原路径继续，复制前缀并接其他候选，增加batch行 | 取选中位置之前的前缀，再调用vLLM续写 |
| 轨迹筛选 | 在配置的前瞻长度比较父子路径并剪枝 | 当前实现扩展若干轮后随机选取所需回答 |
| 主要执行引擎 | HF自定义循环，之后可交给vLLM | 多次vLLM生成调用 |
| 主要参数 | 概率阈值、前瞻长度、差异阈值、路径容量 | 初始数量M、选点数量N、扩展轮数L、每点续写次数T |

`eptree` 的 `get_max_entropy_tokens()` 使用 `-log_prob` 作为选点分数。这是所用 log-prob 条目的负对数概率，不是对完整词表分布求和的 Shannon entropy。理解该实现时，应避免把函数名直接等同于严格的分布熵计算。

复现 LATR 应从 `scripts/*-kt-*.sh` 入手，不能仅因为 `eptree` 也维护树结构，就将两者的采样和剪枝过程混用。

## 15. 效率：哪些工作额外发生了

常规的大批量自回归生成中，vLLM 通常具有更好的吞吐和显存管理效率。它提供专门的推理调度、分页 KV cache 等机制；本项目的 HF 阶段则使用自定义循环，为逐步控制和动态分叉提供了直接的操作空间。

两者都在本地 GPU 上计算，不是“一个本地、一个远程”的区别。LATR 采用 HF 阶段的原因是可以在每轮读取概率、增加和删除路径、直接管理 cache，随后将普通续写交给专门的推理引擎。

这份实现需要考虑的成本包括：

| 成本 | 当前实现中的来源 |
|---|---|
| 更多路径的生成 | 暂时保留多个候选，部分计算最终随剪枝被丢弃 |
| cache复制与筛选 | 分叉时复制父行并拼接，剪枝时用mask筛选各层cache |
| Python与CPU操作 | 分支遍历、`.item()`、`.tolist()`、编辑距离等 |
| 分布式开销 | 使用FSDP actor执行前向，以及退出标记同步等通信 |
| 引擎切换 | sleep／wake、权重同步、垃圾回收和显存整理 |
| 重新prefill | vLLM根据HF返回的前缀建立自己的cache |

一次短 prompt 的 prefill 往往比长回答的整段逐 token decode 更短，不能没有测量就断言重新 prefill 是主要瓶颈。但交接时输入已经包含 HF 生成的前缀：前缀越长、保留路径越多、剩余续写越短，重复 prefill 的相对成本可能越明显。共享前缀的缓存命中情况也会影响结果。

如果分析服务器上的性能，至少应分别计时 HF 树生成、权重同步与切换、vLLM prefill、vLLM decode。仅凭总 rollout 时间无法判断哪项开销最大，也无法给出 Transformers 与 vLLM 固定的速度倍数。

仓库 README 提到可降低 `rollout.micro_batch_size` 缓解 LATR 生成阶段的 OOM，但会牺牲速度。这也与树路径及 cache 的内存占用直接相关。

## 16. 当前实现中需要单独识别的边界与问题

### 16.1 数学符号规则会排除部分比较

`KeyTokenGenConfigMixin` 默认 `keep_math_symbols=True`，随后传给过滤器的 `keep_math_core_symbols`。当前 `RolloutFilter` 会检查父分支比较片段的首个 token，也就是 `sample["b"][0]`，将解码并去除首尾空白后的结果与 `math_core_symbols` 字符串做成员判断，部分样本因此不再参与差异比较。

这是实际代码的判断方式，不应概括成“任何数学分叉都一定被保护”，也不是对整段数学表达式进行语义识别。

### 16.2 数学符号过滤后，样本与分支索引可能错位

在 `RolloutFilter.__call__()` 中，最初建立三个相互对应的列表：

```text
model_batch：比较片段
candidate_ids：这些片段对应的分支ID
candidate_lens：这些片段各自的比较长度
```

数学符号过滤只更新 `model_batch`，没有同步更新另外两个列表；后续却继续使用：

```python
for idx, length, sample in zip(candidate_ids, candidate_lens, model_batch):
    ...
```

例如：

```text
过滤前：
candidate_ids  = [5, 8]
candidate_lens = [20, 30]
model_batch   = [分支5的片段, 分支8的片段]

假设数学符号规则排除了分支5的片段：
candidate_ids  = [5, 8]              # 没有同步过滤
candidate_lens = [20, 30]            # 没有同步过滤
model_batch   = [分支8的片段]

zip得到：分支ID=5、长度=20、比较内容=分支8的片段
```

于是可能用错分母，并将删除结果施加到错误分支。该错位已用原文件的过滤语句构造最小列表示例确认；这不是 GPU 端完整训练验证。本文没有修改这一实现问题，解释剪枝示例时均假定它没有触发。

### 16.3 保留不等于经过全部检查，更不等于正确

EOS跳过比较、数学符号规则、硬上限和路径补齐都会使最终回答不满足“每条路径都实际通过三个差异检查”的直觉描述。路径自身即使通过检查，也可能因祖先被剪而删除。

差异阈值检查默认没有验证答案正确性；完整 rollout 返回后，训练流程才对生成结果计算奖励。轨迹差异、答案正确性和采样数量是三个不同的概念。

## 17. 关键代码与参数索引

### 17.1 按调用顺序阅读代码

| 文件／函数 | 阅读重点 |
|---|---|
| [requirements.txt](requirements.txt)、[readme.md](readme.md) | 版本依赖与`kt`命名 |
| [verl/workers/fsdp_workers.py](verl/workers/fsdp_workers.py)：`_build_rollout()` | 选择`KTRollout`，传入FSDP actor |
| [verl/workers/rollout/kt_rollout.py](verl/workers/rollout/kt_rollout.py)：初始化、`_generate_minibatch()` | HF调用、混合引擎、权重同步、vLLM续写 |
| [model/generation.py](model/generation.py)：`generate()` | 路径初始化、逐token循环、剪枝调用、退出、补齐 |
| [model/generation.py](model/generation.py)：`_sample_new_sequence_with_rollout()` | 候选、容量、增加token行、复制cache |
| [model/sample_filter.py](model/sample_filter.py)：`ProbFilter` | 概率阈值的精确定义 |
| [model/sample_filter.py](model/sample_filter.py)：`RolloutFilter` | 按出生时间比较轨迹、EOS与数学符号规则 |
| [model/utils.py](model/utils.py)：`MixedCache`、`BranchInfo` | cache复制、父子关系、递归删除 |
| [verl/trainer/ppo/ray_trainer.py](verl/trainer/ppo/ray_trainer.py) | rollout返回后计算奖励、进入后续策略更新流程 |
| [model/eptree.py](model/eptree.py) | 对照理解另一种树式rollout，与LATR区分 |

### 17.2 影响生成行为的主要参数

| 参数 | 作用 | 需要区分的概念 |
|---|---|---|
| `rollout.n` | 每个prompt计划返回的回答数N | 不一定全部来自树搜索 |
| `kt.mix_ratio_schedule` | 随训练迭代改变eta，进而确定K | 不是token轮次的调度表 |
| `kt.max_n_branch_per_token` | 单行单位置最多考虑的候选数 | 不是整棵树的容量 |
| `kt.prob_filter_abs_thres` | 非第一候选的概率下界 | 使用严格大于判断 |
| `kt.prob_filter_rel_thres` | 第一候选与其他候选的概率差上界 | 是差值，不是比值 |
| `kt.rollout_filter_steps` | 每条分支从出生位置起的检查长度 | 包含出生token，不是绝对回答位置 |
| `kt.rollout_filter_edit_dist_thres` | 归一化token编辑距离下界 | 衡量轨迹差异，不判断答案正确性 |
| `kt.sample_nk` | 只有一个有效候选时的延伸／回退方式 | 默认`none`，不强制随机采样 |
| `kt.return_nb_thres_init` | 提前退出的数量比例初值 | 不改变每个prompt的K上限 |
| `kt.return_nb_thres_decay` | 随token轮次放宽数量要求 | 不等于降低每个位置的候选数 |
| `kt.force_return_step` | 本阶段循环硬上限 | 达到上限保留现存路径，不补做前瞻检查 |
| `kt.fill_return_sequences` | 数量不足时复制第一条树路径 | 保证数量，不保证多样性 |
| `rollout.kt_mixed_engine` | 默认HF生成后接vLLM续写 | 与eta是两个维度 |
| `rollout.micro_batch_size` | HF阶段一次处理的输入批量规模 | 影响生成显存与效率 |

### 17.3 与源码对应的整体伪代码

下面省略统计、padding、缓存细节和可选消融项，用于串起默认混合模式的执行顺序；数学符号过滤的实际索引问题仍以前文说明为准。

```python
N = 每个prompt的目标回答数
K = round(N * 当前训练阶段的mix_ratio)
O = N - K

if K == 0 or 当前为fallback验证:
    return 使用vLLM普通生成

为每个prompt初始化1条树路径
为每个prompt初始化O条普通采样路径

for step in 本阶段生成范围:
    logits, cache = 当前actor模型对所有路径执行一次前向

    普通路径从处理后的概率分布采样1个token

    对每条未结束的树路径:
        取top-m候选，并判断概率条件

    对每个prompt:
        保证已有路径继续延伸
        按容量K抑制超额新增候选

    各原有树路径接上主候选或按配置回退采样
    其他合格候选复制前缀、mask和cache，追加新行
    为新行记录parent和birth_step=step

    current_step = step + 1
    对达到某个rollout_filter_steps长度的路径:
        取它与父路径从分叉位置开始的片段
        按EOS、数学符号等规则判断是否参与比较
        若参与比较且差异不足，标记删除

    递归删除被标记路径及其后代，并筛选对应cache行
    更新EOS停止记录

    数量够 = 当前rank的有效路径总数达到动态阈值
    年龄够 = 所有保留路径满足step > birth_step + 最长检查窗口
    if 所有rank均满足数量够和年龄够:
        break

# 生成范围耗尽也会到达这里，不额外删除年轻路径。
按prompt整理现存路径
若不足K且开启补齐，复制该prompt的第一条树路径
合并每个prompt的树路径和O条普通路径

if 返回长度已达到回答长度配置:
    return 当前序列

唤醒vLLM并同步当前actor权重
将现存token序列作为vLLM输入，重新建立vLLM自己的cache
使用普通采样生成剩余token，不再执行LATR分叉和差异检查
将续写拼回前缀，返回verl训练流程
```
