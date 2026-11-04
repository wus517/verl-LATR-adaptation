# LATR on verl 0.9.0

> 当前逐 token vLLM 请求执行方案已被用户因效率问题否决。以下保留首版实现与验证记录；最新推荐为引擎内 LATR 调度与共享缓存，见根目录 `docs/latr_engine_design.md`，尚未实施。不要把本目录脚本视为最终训练方案。

目标：Ascend NPU，`Qwen/Qwen3.5-9B`，单轮纯文本 Countdown GRPO。已完成算法和 V1 接口实现，以及 CPU 模拟测试；尚未在 NPU 上完成模型加载、生成或训练验证。

## 实现与参数

入口仍为 `python -m verl.trainer.main_ppo`。设置 `actor_rollout_ref.rollout.latr.enable=true` 后，V1 worker 为每个 prompt 启动一个组生成器，生成 `rollout.n` 条完整回答，再通过原有 reward、TransferQueue、actor/ref logprob、GRPO update 和 CheckpointEngine 流程处理。

树阶段使用 vLLM 的 top raw log probabilities，主路径贪心扩展。绝对阈值 `0.25`、概率差阈值 `0.15`，均采用严格不等式；窗口 `[20,30,50]` 包含分叉 token；归一化编辑距离 `<0.4` 时删除子分支及其后代。容量不足时按备选 token 的 raw probability 全局排序。每组 `round(n * η)` 条 LATR 路径，其余为普通随机采样，默认 `η=0.985**global_step`。树阶段默认最多 512 token，未满组复制根路径，随后各自随机续写。补齐不保证回答唯一。

完整树前缀和续写都进入 response/loss mask；actor 重算原模型的 log probabilities。这不是树选择策略的行为概率，首版禁止 rollout correction/bypass。验证和显式 `__do_sample__=false` 使用原有独立 session 路径。`η=0` 时组生成器执行 n 条普通续写。

首版复用 vLLM 服务，逐 token 请求候选，再显式传入选中的前缀；没有移植旧 HF cache clone。Qwen3.5 的混合注意力/循环状态交给推理后端处理。NPU 脚本先关闭 prefix caching，可能产生大量重复 prefill；当前实现用于验证算法正确性，不代表达到旧实现吞吐。

## 环境准备

在已经可以运行该 checkout 的 Qwen3.5 NPU/FSDP2 训练环境中安装修改后的源码：

```bash
cd verl-0.9.0
uv pip install -e . --no-deps
python -c 'import verl; print(verl.__file__)'
```

确认导入来自本目录。`latr/` 仅作参考，不安装其中的 verl。Ascend 驱动、CANN、torch/torch_npu、vLLM/vllm-ascend 和 Transformers 需采用相互兼容且支持 **Qwen3.5-9B** 的组合；不能仅凭模型家族的 2B 示例推断 9B 已验证。本 checkout 的声明要求 Transformers `>=5.5.3,!=5.6.0,<5.11`、vLLM `>=0.18.0`，不在此替换设备环境的依赖版本。

参考当前源码中的 [Qwen3.5 NPU 示例](../../tests/special_npu/nightly_ci_ascend/run_grpo_qwen3_5_2b_fsdp2_npu.sh) 和 [Ascend 模型文档](../../docs/ascend_tutorial/model_support/examples/qwen3_5_megatron_npu.md)。首版脚本采用 FSDP2、padded/static batch、microbatch=1、禁用 torch compile、参数/优化器 offload。这些是起始配置，不能保证未知型号/显存的 NPU 能运行 9B。

## 数据

已有旧 Countdown 数据时，先写入单独的目录，保留 `data_source`、`reward_model.ground_truth`，将预渲染字符串转换为消息列表：

```bash
python examples/latr/prepare_data.py /path/to/old-countdown ../dataset/countdown-chat --prompt-mode chat
```

`chat` 模式支持旧 `base` 和 `inst-qwen` 模板：提取 user 问题，由 Qwen3.5 自带模板渲染，不重复包裹旧 Assistant 前缀。其他模板会明确报错。已是消息列表的数据保留原样。目录应包含 `train.parquet` 和 `test.parquet`。自备数据也可直接采用以下结构：

```python
{
    "data_source": "countdown",
    "prompt": [
        {
            "role": "user",
            "content": "Using the numbers [1, 2, 3], create an equation that equals 9. Use each number exactly once and only +, -, *, /. Put the final expression in <answer> </answer> tags on the last line.",
        }
    ],
    "reward_model": {"style": "rule", "ground_truth": {"numbers": [1, 2, 3], "target": 9}},
}
```

`reward.py` 接受新版 response-only 接口，以受限 AST 求值替代 eval；保留旧最后一行 `<answer>` 提取与 `0/0.1/1` 奖励语义，因此最后一行之后多出的换行也会影响得分。当前脚本只配置 Countdown reward，未接入 Math/DAPO。

如需对照旧 base 模型提示词，使用 `--prompt-mode preformatted` 转换至 `countdown-v1`，再使用 `run_grpo_countdown.sh`。它使用透传模板；不要将该模板直接用于 Qwen3.5 chat 实验。

## NPU 验证顺序

以下均在 `verl-0.9.0/` 目录运行。先设置本机路径、NPU 数量和推理 TP。示例中的 4 卡和 TP=2 只演示参数传递，需按实际硬件设置；NPU 数量必须能被 TP 整除，训练 batch 也必须满足框架的数据并行约束。

```bash
export MODEL_PATH=/path/to/Qwen3.5-9B
export DATA_DIR=/path/to/countdown-chat
export NDEVICES_PER_NODE=4
export ROLLOUT_TP=2
```

1. 验证设备后端能给出 raw top-2，且能从显式分叉前缀继续生成：

   ```bash
   python examples/latr/probe_vllm.py --model "$MODEL_PATH" --tensor-parallel-size "$ROLLOUT_TP"
   ```

   probe 比较两种温度下的 raw 概率，再测试两条前缀续写。失败时先解决 vLLM Ascend/模型能力问题；它不验证训练、TQ 或权重同步。

2. 普通 GRPO baseline，先确认模型训练链可运行：

   ```bash
   TOTAL_STEPS=2 bash examples/latr/run_grpo_qwen35_9b_npu.sh \
       actor_rollout_ref.rollout.latr.enable=false trainer.test_freq=-1
   ```

3. 短树 LATR smoke test，至少两步以覆盖 actor update 后的下一轮 rollout：

   ```bash
   TOTAL_STEPS=2 ROLLOUT_N=2 bash examples/latr/run_grpo_qwen35_9b_npu.sh \
       actor_rollout_ref.rollout.latr.force_return_step=32 trainer.test_freq=-1
   ```

   32 token 树预算不足以完整覆盖 50-token 窗口，仅用于链路验证。检查完整组数量、finite loss/reward、无模型版本混用、每步权重同步和下一步生成。训练统计为每条 session 的 `extra_fields.latr`；尚未加入 console 聚合指标。

4. 恢复论文窗口及树预算，运行训练和普通采样验证：

   ```bash
   TOTAL_STEPS=2 bash examples/latr/run_grpo_qwen35_9b_npu.sh
   ```

   然后再做保存/恢复、batch/显存调优、Math/DAPO 和性能对比。当前 smoke 配置 `save_freq=-1`，不验证恢复。

## 本地检查

```bash
python -m unittest discover -s tests/experimental/latr -p 'test_*.py' -v
bash -n examples/latr/run_grpo_countdown.sh
bash -n examples/latr/run_grpo_qwen35_9b_npu.sh
ruff check verl/experimental/latr examples/latr tests/experimental/latr
```

本次本地结果：31 个测试全部通过（包含 Hydra 用例，无跳过），Ruff、`bash -n`、`git diff --check` 通过。

核心测试不需要 torch/Ray/vLLM；安装 hydra-core 后额外运行真实 YAML 组合、实际 shell 参数及配置拒绝测试，否则该用例跳过。框架边界测试用模拟服务执行真实方法，不代表真实 Ray、TransferQueue、设备 kernel 或训练通过。

仅支持 V1 synchronous trainer + asynchronous built-in vLLM service + 单轮文本。多模态、工具调用、连续 token、PD、speculative rollout、自定义 manager/server、完全异步训练尚不支持，配置会提前拒绝主要不兼容组合。LATR 默认关闭。
