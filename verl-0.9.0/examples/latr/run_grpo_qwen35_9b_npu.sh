#!/usr/bin/env bash
# Initial NPU configuration, based on the checkout's Qwen3.5 FSDP2 examples.
# Real NPU memory switching and Qwen3.5 cached forward require device validation.
set -euo pipefail

LATR_EXAMPLE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${NDEVICES_PER_NODE:?Set NDEVICES_PER_NODE to the available NPU count}"
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
export N_GPUS=$NDEVICES_PER_NODE
export DATA_DIR=${DATA_DIR:-../dataset/countdown-chat}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
if (( NDEVICES_PER_NODE < 1 || ROLLOUT_TP < 1 || NDEVICES_PER_NODE % ROLLOUT_TP != 0 )); then
    echo "NDEVICES_PER_NODE must be positive and divisible by ROLLOUT_TP" >&2
    exit 1
fi
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Qwen3.5 Gated Delta Net follows the existing padded, static-batch NPU path.
# Disable prefix caching initially: hybrid recurrent-state caching is backend/version dependent.
bash "$LATR_EXAMPLE_DIR/run_grpo_countdown.sh" \
    trainer.device=npu \
    trainer.experiment_name=latr-qwen35-9b-npu \
    actor_rollout_ref.model.custom_chat_template=null \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$NDEVICES_PER_NODE" \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.offload_policy=true \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.latr.tree_micro_batch_size=1 \
    "$@"
