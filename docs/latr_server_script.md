# token RPC 分支的服务器 LATR 脚本

脚本：[latr.sh](../latr.sh)。适用分支：`archive/latr-vllm-token-rpc`。模型和设备沿用现有服务器配置：Qwen3.5-9B、Ascend NPU、FSDP2、vLLM TP=2。该版本的树阶段仍逐 token 请求 vLLM，未引入 hybrid 分支的 HF 缓存树引擎。

## 脚本来源与启动

本分支没有根目录 `grpo.sh`，因此从 `feat/latr-hybrid-cached` 的 `latr.sh` 中剥离此前新增的 LATR 块，恢复 486 行 GRPO 底稿。保留这些原有行的全部内容、注释、空行、格式与启动流程，只插入当前分支需要的内容。

所有节点上传同一分支的代码，在原来的启动位置运行：

```bash
EXP_NAME=qwen35_9b_latr_rpc_run1 bash latr.sh
```

沿用原脚本的 `lhx_sh`、`lhx_save`、ModelArts/多节点环境变量、数据集和自定义奖励函数路径，以及 CANN/ATB、依赖安装、代理和 Ray 流程。输出目录仍保留原来的 `grpo` 命名，使用不同的 `EXP_NAME` 区分实验。

源码默认位于 `/cache/algorithm/verl-0.9.0`，也识别源码直接位于 `/cache/algorithm` 的布局。其他布局在所有节点显式设置：

```bash
export VERL_DIR=/absolute/path/to/verl-0.9.0
EXP_NAME=qwen35_9b_latr_rpc_run1 bash latr.sh
```

新增块检查当前分支的 `verl/experimental/latr/backend.py`，在 Ray 启动前设置 `PYTHONPATH` 并进入源码目录。

## 对齐论文 Math 的默认参数

依据 [论文参数记录](latr_algorithm.md#9-论文实验参数) 的附录 B.3。为了保持原脚本行不变，部分默认值在原赋值前新增设置，其余作为训练命令末尾的覆盖项。

| 项目 | 当前脚本默认值 | 来源或说明 |
| --- | --- | --- |
| 每个 prompt 的回答数 | 8 | 论文 |
| 分叉绝对概率 / 概率差阈值 | 0.25 / 0.15 | 论文；严格不等式 |
| 前瞻窗口 / 编辑距离阈值 | [20,30,50] / 0.4 | 论文 |
| 初始混合比例 / 指数衰减率 | 1.0 / 0.995 | 论文 Math，`η(t)=0.995^t` |
| train batch / PPO mini batch | 256 / 256 | 论文；mini batch 指 prompt 数，训练轨迹数还乘 n |
| 最大回答长度 | 8192 | 论文 Math |
| 学习率 / clip ratio | 1e-6 / 0.2 | 论文，原脚本已一致 |
| KL loss 系数 | 0.01 | 论文附录 |
| 训练 temperature / top-k / top-p | 1.0 / -1 / 1.0 | 论文 |
| 验证 temperature / top-k / top-p | 0.6 / 20 / 0.95 | 论文；启用采样，验证 n=8 |
| 训练步数 | 500 | 论文；提供足够 epoch，以步数上限结束 |
| loss aggregation | token-mean | 显式按有效 token 归一化，对应论文损失公式 |

后缀匹配、ROUGE-L、数学符号保护关闭；不设置分段混合比例；饱和阈值为 1，不随树深度降低。

这不是完整的论文复现实验：模型、设备、数据、奖励和 prompt 模板仍使用现有服务器配置，算法训练入口为 GRPO，未启用 DAPO 组过滤。每次候选上限为 2（包含主候选），是当前实现沿用旧代码的限制。训练 micro batch 默认 1，论文 Math 为 4；通过累积实现设定的 mini batch，以控制 NPU 峰值内存。

## 工程参数与覆盖

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 256 / 256 | 外部指定时优先于论文默认值 |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 1500 / 8192 | prompt 沿用原脚本，response 对齐论文 |
| `MAX_MODEL_LEN` | prompt + response，默认 9692 | 覆盖完整上下文 |
| `MAX_NUM_BATCHED_TOKENS` | MAX_MODEL_LEN | vLLM 调度预算，保持原 chunked prefill 设置 |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | MAX_MODEL_LEN | 保持相关 token 预算一致；静态训练仍由 micro batch 决定 |
| `TOTAL_STEPS` | 500 | 实际步数上限 |
| `TOTAL_EPOCHS` | TOTAL_STEPS | 为循环数据提供足够 epoch，可按需覆盖 |
| `LATR_TRAIN_MICRO_BATCH_SIZE` | 1 | actor 每卡 micro batch |
| `LATR_KL_LOSS_COEF` | 0.01 | KL loss 系数 |
| `LATR_VAL_N` | 8 | 每个验证 prompt 的独立采样数 |
| `LATR_TREE_MAX_TOKENS` | -1 | 不额外强制截短树；按饱和、EOS、回答长度退出 |
| `LATR_MAX_CONCURRENT_REQUESTS` | 64 | 每个 agent worker 的 vLLM 请求并发上限，不是全局上限 |

设置树上限属于工程折中，会改变转入普通采样的时机；默认不沿用旧实现的 512/1024 token 强制上限。逐 token RPC 在长树上可能耗时较大。8192 长度和验证 n=8 也会增加真实资源开销，尚无该服务器上的实测结果。

例如短运行检查（仍使用完整窗口，树上限是显式折中）：

```bash
EXP_NAME=qwen35_9b_latr_rpc_smoke TOTAL_STEPS=2 \
TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH_SIZE=16 MAX_RESPONSE_LENGTH=2048 \
LATR_TREE_MAX_TOKENS=1024 LATR_MAX_CONCURRENT_REQUESTS=16 \
bash latr.sh
```

模型长度和 token 预算会跟随回答长度重新计算；若自行指定了 `MAX_MODEL_LEN` 等变量，需保证与长度兼容。

## 迁移的训练优化

从 hybrid 分支的提交 `710de92` 迁移三处 `grad_sync_every_micro_batch` 改动：

1. `verl/trainer/config/engine/fsdp.yaml` 注册配置，通用默认 false。
2. `verl/workers/config/engine.py` 在 `FSDPEngineConfig` 定义字段。
3. `verl/workers/engine/fsdp/transformer_impl.py` 开启时不进入梯度 `no_sync`，每个 micro batch 同步梯度。

原脚本已有 `grad_sync_every_micro_batch=True`，现在可直接生效，不添加 `+` 注册或 `~` 删除。梯度继续累积，mini batch 末尾才更新优化器；用更多通信降低完整梯度驻留内存。配置与逻辑均与 hybrid 分支的这三处修改一致，不迁移 hybrid 专用的权重导出或树执行代码。

## 验证边界

本地结果：

- GRPO 底稿全部 486 行逐字节保留，新脚本只有插入内容。
- `bash -n latr.sh`、修改的 Python 文件 Ruff 检查、`git diff --check` 通过。
- 从实际脚本隔离展开训练参数，论文默认配置与短运行环境覆盖均通过 Hydra/LATR/verl 校验；actor/ref/rollout dataclass 构建通过。确认 actor 引擎实际读到 `grad_sync_every_micro_batch=True`、衰减率为 `0.995`，且没有 hybrid 专属参数。
- 梯度同步与模型输出回归测试、原有 LATR 测试合计 **40 项通过**。新增测试覆盖开关 true/false 时的同步选择和共享参数梯度累积结果。
- **1 项双进程 FSDP 梯度等价性测试未通过运行环境初始化**：本机 Gloo 地址解析失败；指定 `lo0` 后，沙箱禁止 `uv_bind`。申请运行沙箱外本地测试时，自动审批服务返回 `404 model not found`，未执行该次命令。此项没有得到梯度等价性的实测结果，需在允许本机通信的环境运行。

NPU 多节点训练、raw top-logprob 后端能力、吞吐和峰值内存仍需服务器验证；当前脚本保持原 `enforce_eager=False`、`enable_chunked_prefill=True`、`enable_prefix_caching=False`。
