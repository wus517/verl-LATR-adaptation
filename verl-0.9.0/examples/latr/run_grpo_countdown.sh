#!/usr/bin/env bash
set -euo pipefail

LATR_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$LATR_REPO_ROOT"

# Install this modified checkout first; do not install the old latr/verl package.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B}
DATA_DIR=${DATA_DIR:-../dataset/countdown-v1}
N_GPUS=${N_GPUS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
TOTAL_STEPS=${TOTAL_STEPS:-2}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export TOKENIZERS_PARALLELISM=false

# Raw prompts from the old dataset already contain the base-model prompt template.
LATR_CHAT_TEMPLATE="{% for message in messages %}{{ message['content'] }}{% endfor %}"

python -m verl.trainer.main_ppo \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.total_epochs=100 \
    trainer.val_before_train=false \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.logger=console \
    trainer.project_name=latr \
    trainer.experiment_name=grpo-countdown-v1 \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.gen_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.truncation=error \
    data.filter_overlong_prompts=true \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    "actor_rollout_ref.model.custom_chat_template=\"$LATR_CHAT_TEMPLATE\"" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.latr.enable=true \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    reward.custom_reward_function.path="$LATR_REPO_ROOT/examples/latr/reward.py" \
    reward.custom_reward_function.name=compute_score \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    "$@"
