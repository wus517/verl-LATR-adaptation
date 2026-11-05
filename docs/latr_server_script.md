# 服务器 LATR 启动脚本

更新：2026-11-05。脚本：[根目录 latr.sh](../latr.sh)。依据 `grpo.sh` 的有效 shell 命令与参数适配，未使用其注释里的模型规模、环境版本或任务描述。`grpo.sh` 保留原样。

## 启动

将当前 `feat/latr-hybrid-cached` 的代码和 `latr.sh` 一起上传到原先服务器的代码目录，所有节点保持相同代码、路径与 Python 环境。与原脚本相同，默认将 `/opt/huawei/schedule-train/algorithm/` 的内容复制到 `/cache/algorithm/`。

平台已提供 `lhx_sh`、`lhx_save`、`MA_NUM_HOSTS`、`MA_CURRENT_IP`、`VC_TASK_INDEX`、`VC_WORKER_HOSTS` 时，在原来每个节点的启动位置改用：

```bash
EXP_NAME=qwen35_9b_latr_run1 bash latr.sh
```

仅 rank 0 提交训练；其他节点注册到 Ray head 后保持运行。脚本会等待所有节点的 NPU 资源到齐，默认等待上限 1800 秒。

路径与平台变量不同可显式指定：

```bash
export BASE_PATH=/path/to/data_root
export SAVE_PATH=/path/to/output_root
export VERL_DIR=/cache/algorithm/verl-0.9.0
export EXP_NAME=qwen35_9b_latr_run1
bash latr.sh
```

`BASE_PATH` 默认取 `lhx_sh`，`SAVE_PATH` 默认取 `lhx_save`。`VERL_DIR` 必须是当前适配版源码目录，不能是旧参考代码 `latr/` 或服务器原来的未修改 verl。默认自动识别 `/cache/algorithm/verl-0.9.0` 和源码直接放在 `/cache/algorithm` 两种布局，启动前核验实际导入来源。每个节点在 Ray 启动前配置同一个 `PYTHONPATH`。

默认数据和奖励：

| 变量 | 默认路径 |
| --- | --- |
| MODEL_PATH | `${BASE_PATH}/model/Qwen3_5_9B` |
| TRAIN_FILE | `${BASE_PATH}/xyi/train_verl.json` |
| VAL_FILE_HIGH | `${BASE_PATH}/xyi/high_pass_val_verl.json` |
| VAL_FILE_LOW | `${BASE_PATH}/xyi/low_pass_val_verl.json` |
| REWARD_FILE | `${BASE_PATH}/xyi/verify_choice_answer_new_v4.py` |

沿用原脚本的 CANN/ATB source、代理、NPU/HCCL 环境，以及 accelerate、aiohttp、aiofiles、math-verify 安装。已有依赖且不需要联网安装时设置 `INSTALL_RUNTIME_DEPS=0`。不会安装或替换 torch、torch_npu、Transformers、vLLM、vLLM-Ascend。匹配环境与适配版 verl 的其他依赖仍需预先准备。

## 参数变化

保留有效原设置：训练 batch 128、mini batch 32、n=8、prompt 1500、response 3072、模型长度 4800、lr=1e-6、temperature=1、top_p=.99、rollout TP=2、vLLM memory utilization=.7、KL loss、一次 epoch、原有两个验证集与自定义奖励、`enable_thinking=false`。

| 配置 | 适配结果 | 原因 |
| --- | --- | --- |
| trainer.use_v1 / trainer_mode | true / sync | 当前混合路线入口 |
| data.gen_batch_size | 与 train_batch_size 相同 | 一整批完成树阶段后再切 vLLM |
| actor.fsdp_config.optimizer_offload | true | 为树阶段腾出优化器设备内存；当前实现的必需配置 |
| ref.fsdp_config.offload_policy | true | reference 也由 FSDP2 CPU offload 管理 |
| rollout.calculate_log_probs | false | actor 重算整条轨迹的模型 logprob |
| rollout.latr.enable | true | 启用 LATR |
| tree worker / microbatch | 1 / 1 | 首次运行控制模型副本和树缓存；分支仍批量执行 |
| tree force_return_step | 512 | 当前 LATR 默认树长度上限 |
| enforce_eager / chunked_prefill | true / false | 采用当前 NPU 混合路线的初始配置 |
| grad_sync_every_micro_batch | 移除 | 本仓库配置没有该字段 |
| forward_prefetch | 使用默认 false | 原字段只用于 FSDP1，本脚本使用 FSDP2 |
| rollout.engine_kwargs.chat_template_kwargs | 移除 | token prompt 已由 data 的模板参数生成，不向 vLLM engine 传额外无效配置 |
| 输出目录 | `grpo` 改为 `latr` | 隔离 GRPO 的日志、历史与 checkpoint |

默认 LATR 概率阈值 .25/.15、窗口 [20,30,50]、编辑距离阈值 .4、eta=1*.985^global_step。没有因任务名称推断或启用数学符号保护等可选规则。

修正了原脚本的重复 max_num_batched_tokens 参数、未提供 SAVELOG_FILE_NAME 的情况；Ray 用 Python API 检查资源，避免依赖 `ray status` 的文本格式；训练失败直接返回真实退出码。

## 预览与小规模运行

本地只预览（不复制到 /cache、不装依赖、不加载设备、不启动 Ray）：

```bash
LATR_DRY_RUN=1 BASE_PATH=/server/data SAVE_PATH=/server/save NNODES=2 \
  bash latr.sh > /tmp/latr_args.txt
```

服务器前两步试跑，所有节点使用一致设置；训练 batch/mini batch 可按实际设备数调整：

```bash
EXP_NAME=qwen35_9b_latr_smoke TOTAL_STEPS=2 SAVE_FREQ=-1 VAL_FREQ=-1 \
  bash latr.sh
```

命令尾部支持 Hydra 覆盖，例如：

```bash
EXP_NAME=qwen35_9b_latr_run2 bash latr.sh \
  actor_rollout_ref.rollout.latr.tree_micro_batch_size=2
```

也可设置 `LATR_TREE_WORKERS`、`LATR_TREE_MICRO_BATCH_SIZE`、`LATR_TREE_MAX_TOKENS`、`LATR_WEIGHT_BUCKET_MB`。树 worker 选择全局 rank 0..count-1，不自动跨节点均匀分布。

## 验证与边界

已在本地检查 `bash -n`，并用 dry-run 的实际参数通过 Hydra compose、LATR 配置校验、verl 通用配置校验与 rollout/actor/ref dataclass 构建（ref 按原 worker 的字段转换步骤处理）。未执行服务器 NPU 训练。

当前树阶段只支持纯文本；原配置里的 `data.image_key=images` 保留，但真实包含图像/视频/音频的样本不受当前 LATR 后端支持。树阶段需要一个设备容纳完整 9B 模型和分支缓存，**不使用 vLLM 的 TP=2**。具体 NPU 型号/容量、缓存峰值、vLLM sleep 释放量与吞吐仍需实机验证。
