# LATR：HF 缓存树 + 原版 vLLM 续写

实现设计与验证边界见 [设计文档](../../../docs/latr_engine_design.md)。开关是 `actor_rollout_ref.rollout.latr.enable=true`，默认关闭。旧逐 token RPC 实现在根仓库 `archive/latr-vllm-token-rpc` 分支。

## 环境与数据

安装本地修改后的 `verl-0.9.0`，不要安装参考目录 `latr/verl`。使用与 NPU 平台匹配的 torch_npu、CANN、vLLM-Ascend；本机尚未做 NPU 实测。CPU 用例使用 Transformers 5.9.0；其他声明支持版本的缓存契约仍需验证。

旧 Countdown parquet 可以转换为新版聊天消息格式，输出到新目录：

```bash
cd verl-0.9.0
python examples/latr/prepare_data.py /path/to/old-countdown /path/to/countdown-chat --prompt-mode chat
```

`chat` 模式提取旧预格式化 prompt 的用户正文并使用模型自己的模板。`preformatted` 模式保留旧文本，配套通用脚本采用拼接 content 的模板。奖励函数在 `reward.py`，严格限制表达式节点并核验数字使用。

## NPU Qwen3.5-9B 启动

```bash
NDEVICES_PER_NODE=8 \
MODEL_PATH=/path/to/Qwen3.5-9B \
DATA_DIR=/path/to/countdown-chat \
ROLLOUT_TP=2 TOTAL_STEPS=2 \
bash examples/latr/run_grpo_qwen35_9b_npu.sh
```

脚本显式要求设备数量；上面的 8 只是命令示例。需要选择真实容量。树模型使用一个完整设备，**不会继承 vLLM TP=2**；9B BF16 权重约 18 GB，另需激活、缓存和临时重排空间。训练使用 FSDP2 CPU offload，主存还需容纳额外 HF 副本、训练/优化器参数和权重传输。

关键调参：

```bash
# microbatch 为 prompt 数，每个 prompt 内的树分支仍一起 forward
... bash examples/latr/run_grpo_qwen35_9b_npu.sh \
  actor_rollout_ref.rollout.latr.tree_micro_batch_size=2 \
  actor_rollout_ref.rollout.latr.tree_worker_count=2
```

`tree_worker_count` 默认 1，选择全局训练 rank 0..count-1，各占其所在设备；不自动均匀分布到多节点。先测单 worker，增大前确认设备与主存容量。

每轮整批树生成后才切换 vLLM，不在 microbatch 间反复切换。每条路径最多请求一次 vLLM 续写。NPU 原版 sleep 是否完全释放内存、HF torch DeltaNet fallback 吞吐与实际峰值要在训练机测量，不能仅据 microbatch=1 推断不会 OOM。

## 检查

```bash
python -m pytest -q tests/experimental/latr/test_hybrid_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py
bash -n examples/latr/run_grpo_countdown.sh
bash -n examples/latr/run_grpo_qwen35_9b_npu.sh
```

树算法与窗口、KV/recurrent 缓存重排已用真实小 Llama/Qwen3.5 在 CPU 验证。完整 Ray/FSDP2/vLLM-Ascend 两步训练及恢复未在本机验证。首次实机运行应保留日志，观察 `timing_s/latr_prepare`（具体 logger 前缀随 verl 统计格式）、轨迹 extra_fields 中的树统计和版本错误。
