# 服务器 LATR 启动脚本

更新：2026-11-05。脚本：[latr.sh](../latr.sh)。以 `grpo.sh` 为原样底稿，只插入 LATR 所需内容；原文件不修改。原有每一行（包括注释、空行、格式、重复参数）均保留，算法与服务器设置以有效命令为依据。

## 启动

将当前 `feat/latr-hybrid-cached` 分支的代码和脚本上传到服务器，各节点使用相同源码和 Python 环境。在原有启动位置将 `bash grpo.sh` 换为：

```bash
EXP_NAME=qwen35_9b_latr_run1 bash latr.sh
```

沿用原脚本所需的平台变量、数据路径、模型路径、依赖安装、CANN/ATB、代理及 Ray 启动流程。默认仍把 `/opt/huawei/schedule-train/algorithm` 复制到 `/cache/algorithm`。原有 `grpo` 输出目录命名也保留，可用不同的 `EXP_NAME` 区分实验。

新增的源码定位块默认使用 `/cache/algorithm/verl-0.9.0`，也支持源码直接位于 `/cache/algorithm`。其他布局在每个节点设置：

```bash
export VERL_DIR=/absolute/path/to/verl-0.9.0
EXP_NAME=qwen35_9b_latr_run1 bash latr.sh
```

`VERL_DIR` 必须指向本分支含 `verl/experimental/latr/hybrid.py` 的源码目录。新增代码在 Ray 启动前设置 `PYTHONPATH` 并进入该目录，确保训练使用适配版。

## 追加的参数

原训练参数行保持原样，在命令末尾追加以下覆盖项。重复配置以最后一次赋值为准。

| 参数 | 最终值 | 用途 |
| --- | --- | --- |
| `trainer.use_v1` | `true` | 使用 V1 训练入口 |
| `trainer.v1.trainer_mode` | `sync` | 使用当前混合 LATR 支持的同步模式 |
| `data.gen_batch_size` | `${TRAIN_BATCH_SIZE}` | 按训练批次完成树阶段，再切换 vLLM |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `true` | 树阶段释放优化器设备内存 |
| `actor_rollout_ref.ref.fsdp_config.offload_policy` | `true` | reference 使用 FSDP2 CPU offload |
| `actor_rollout_ref.rollout.calculate_log_probs` | `false` | 由 actor 重算完整轨迹 logprob |
| `actor_rollout_ref.rollout.latr.enable` | `true` | 开启混合 LATR |
| `actor_rollout_ref.rollout.latr.mix_ratio_decay` | `0.995` | 对齐论文 Math 的混合比例衰减率 |

另显式传入四个树阶段默认值，可通过环境变量调整：

| 环境变量 | 默认值 | LATR 配置字段 |
| --- | --- | --- |
| `LATR_TREE_WORKERS` | `1` | `tree_worker_count` |
| `LATR_TREE_MICRO_BATCH_SIZE` | `1` | `tree_micro_batch_size` |
| `LATR_TREE_MAX_TOKENS` | `512` | `force_return_step` |
| `LATR_WEIGHT_BUCKET_MB` | `128` | `weight_bucket_mb` |

概率阈值、剪枝窗口、初始混合比例等沿用当前 `LATRConfig` 的默认值。衰减率显式覆盖为论文 Math 的 `0.995`，即 `η(t)=1.0×0.995^t`；通用配置中的 Countdown 默认值 `0.985` 不变。原 `enforce_eager=False`、`enable_chunked_prefill=True`、KL 系数以及训练和输出设置全部保留，因此本脚本并非完整的论文实验参数复现。

### 原参数的兼容处理

原脚本有 `actor_rollout_ref.actor.fsdp_config.grad_sync_every_micro_batch=True`，本仓库的 verl 0.9.0 YAML 与 dataclass 均没有该字段。为了保留原行，同时避免 Hydra 和 dataclass 报错，额外在命令开头用 `+` 注册此键，末尾用 `~` 移除；最终训练配置不包含这个不受支持的字段。这只是启动参数兼容，不实现该选项的旧版语义。

## 验证与运行边界

本地已验证：删除新增块后可逐字节还原 `grpo.sh`；`bash -n latr.sh` 通过。训练命令的参数在隔离 shell 中展开后，通过 Hydra 配置合成、LATR 与 verl 配置校验，以及 rollout、actor、reference 的 dataclass 构建。没有执行原脚本的环境安装、Ray 启动或 NPU 训练。

当前树阶段支持纯文本，单个树 worker 需要单张 NPU 容纳完整 9B 模型及分支缓存；原脚本的 vLLM `TP=2` 只作用于续写阶段。实际显存和吞吐需在服务器验证。
