# LATR 适配 verl 0.9.0：实施状态与计划

更新：2026-11-04。

目标是在本地 `verl-0.9.0/` 实现论文 LATR，旧 `latr/` 仅作参考。目标训练环境为 Ascend NPU，模型 Qwen3.5-9B。

配套文档：[算法](latr_algorithm.md)、[旧代码](latr_code.md)、[当前混合实现设计与限制](latr_engine_design.md)、[运行说明](../verl-0.9.0/examples/latr/README.md)。

## Git 状态

根目录独立仓库已经建立，源码和论文作为普通文件管理。`main`/初始基线为 `290cdd1`。

- `archive/latr-vllm-token-rpc`：保存先前逐 token 请求 vLLM 的实现和讨论文档。
- `feat/latr-hybrid-cached`：当前混合实现开发分支，从初始基线建立。
- 当前改动保留在工作区，尚未提交；不使用旧 `latr/` 的 Git 历史。

## 已完成

- 论文、旧代码、目标新版 V1/TransferQueue/FSDP2/vLLM 生命周期调研。
- 明确默认概率阈值、包含分叉 token 的比较窗口、候选概率排序、稳定 ID、递归剪枝、EOS 和不足组复制。
- 带缓存的 HF 批量树执行；不逐 token 调用 vLLM/Ray；Qwen3.5 KV、卷积/recurrent 缓存一起分叉。
- 独立树进程与训练 monkey patch 隔离；CPU 侧常驻模型，设备阶段错开。
- 最新 FSDP2 全量参数分桶导出、覆盖检查、训练与续写版本一致性检查。
- 整批树完成并卸载后，再同步并唤醒 stock vLLM，按路径普通续写一次。
- V1 sync 和 AgentLoop/TransferQueue 接通；完整响应 mask；actor 重算 logprob。
- validation/显式 greedy 走普通采样；零树比例跳过 HF 导出。
- 提供 Countdown 数据转换、奖励、通用/NPU 训练脚本。
- CPU 真模型缓存正确性与阶段/错误路径测试。

## 当前范围与实际限制

首版限定 V1 sync + FSDP2 offload + GRPO + dense BF16 文本。具体不支持组合在入口明确报错，见设计文档。单个树 worker 要完整容纳 HF 模型；vLLM 可以 TP，树阶段没有 TP。默认一个树 worker、每批一个 prompt（该 prompt 的分支批量执行），可调整。

HF 树缓存不转交 vLLM；续写时重新 prefill 一次。每轮有额外 actor 导出、模型传输和 CPU RAM 开销。NPU Qwen3.5 树推理目前使用原生 torch DeltaNet fallback，融合内核优化尚未实现。

这份实现已经接通代码路径并通过 CPU 检查，**不等于已经完成 NPU 完整训练或性能复现**。

## 接下来按顺序验收

1. 在目标 NPU 安装本地 `verl-0.9.0`，核对实际 checkpoint 和依赖，确认单卡能容纳树模型+缓存。
2. 最小生成验证：EOS、前缀/后缀长度、n、窗口、版本与内存释放。
3. 两步以上 GRPO 更新验证：真实参数更新、奖励、advantage、loss、梯度、验证集采样。
4. 保存/恢复验证及真实失败场景清理。
5. 性能比较与 profile 后，决定是否增加树 worker/microbatch 或实现 NPU DeltaNet 融合算子；不先改算法换取性能。
6. 以上完成后再扩展 Math、DAPO、更多执行后端或树阶段模型并行。
