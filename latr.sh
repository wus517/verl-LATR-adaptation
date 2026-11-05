#!/usr/bin/env bash
# Run this script on every server node with the same environment variables.
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
LATR_DRY_RUN=${LATR_DRY_RUN:-0}
PROJECT_ROOT=${PROJECT_ROOT:-/cache/algorithm}
SOURCE_ROOT=${SOURCE_ROOT:-/opt/huawei/schedule-train/algorithm}
CANN_ENV=${CANN_ENV:-/home/ma-user/Ascend/cann-8.5.2/set_env.sh}
ATB_ENV=${ATB_ENV:-/home/ma-user/Ascend/nnal/atb/set_env.sh}

if [[ "$LATR_DRY_RUN" != 1 ]]; then
    PROXY_ADDR=${PROXY_ADDR:-http://10.31.196.119:80}
    export http_proxy="$PROXY_ADDR" https_proxy="$PROXY_ADDR"
    export HTTP_PROXY="$PROXY_ADDR" HTTPS_PROXY="$PROXY_ADDR"
    export ftp_proxy="$PROXY_ADDR" FTP_PROXY="$PROXY_ADDR"
    export LD_PRELOAD="/home/ma-user/Ascend/cann-8.5.2/lib64/libprofapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
    if [[ "$SOURCE_ROOT" != "$PROJECT_ROOT" ]]; then
        mkdir -p "$PROJECT_ROOT"
        cp -a "$SOURCE_ROOT/." "$PROJECT_ROOT/"
    fi
    source "$CANN_ENV"
    source "$ATB_ENV"
fi

# VERL_DIR must point to this branch's verl-0.9.0, identically on all nodes.
if [[ -z "${VERL_DIR:-}" ]]; then
    if [[ "$LATR_DRY_RUN" == 1 ]]; then
        VERL_DIR="$SCRIPT_DIR/verl-0.9.0"
    elif [[ -f "$PROJECT_ROOT/verl-0.9.0/verl/experimental/latr/hybrid.py" ]]; then
        VERL_DIR="$PROJECT_ROOT/verl-0.9.0"
    elif [[ -f "$PROJECT_ROOT/verl/experimental/latr/hybrid.py" ]]; then
        VERL_DIR="$PROJECT_ROOT"
    else
        echo "Cannot find adapted verl. Upload this branch and set VERL_DIR to its verl-0.9.0 directory." >&2
        exit 1
    fi
fi
[[ -f "$VERL_DIR/verl/experimental/latr/hybrid.py" ]] || { echo "Invalid VERL_DIR: $VERL_DIR" >&2; exit 1; }
VERL_DIR=$(cd "$VERL_DIR" && pwd)
export PYTHONPATH="$VERL_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$VERL_DIR"

BASE_PATH=${BASE_PATH:-${lhx_sh:-}}
SAVE_PATH=${SAVE_PATH:-${lhx_save:-}}
: "${BASE_PATH:?Set BASE_PATH or lhx_sh}"
: "${SAVE_PATH:?Set SAVE_PATH or lhx_save}"
XYI_DIR="$BASE_PATH/xyi"
TRAIN_FILE=${TRAIN_FILE:-$XYI_DIR/train_verl.json}
VAL_FILE_HIGH=${VAL_FILE_HIGH:-$XYI_DIR/high_pass_val_verl.json}
VAL_FILE_LOW=${VAL_FILE_LOW:-$XYI_DIR/low_pass_val_verl.json}
REWARD_FILE=${REWARD_FILE:-$XYI_DIR/verify_choice_answer_new_v4.py}
MODEL_PATH=${MODEL_PATH:-$BASE_PATH/model/Qwen3_5_9B}
NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
NPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-${MA_NUM_GPUS:-8}}
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
MASTER_ADDR=${MASTER_ADDR:-${VC_WORKER_HOSTS%%,*}}
MASTER_ADDR=${MASTER_ADDR:-${MA_CURRENT_IP:-127.0.0.1}}
NODE_IP=${NODE_IP:-${MA_CURRENT_IP:-}}
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
RAY_READY_TIMEOUT=${RAY_READY_TIMEOUT:-1800}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-qwen35_9b_latr}}
PROJECT_NAME=${PROJECT_NAME:-verl_latr_math1w}
TIMESTAMP=${RUN_TIMESTAMP:-$(date +'%Y%m%d_%H%M%S')}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$SAVE_PATH/tensorboard/latr/${EXPERIMENT_NAME}_${TIMESTAMP}}
ROLLOUT_HISTORY_DIR=${ROLLOUT_HISTORY_DIR:-$SAVE_PATH/rollout_history/latr/${EXPERIMENT_NAME}_${TIMESTAMP}}
TRAIN_ROLLOUT_DIR=${TRAIN_ROLLOUT_DIR:-$ROLLOUT_HISTORY_DIR/train}
VAL_ROLLOUT_DIR=${VAL_ROLLOUT_DIR:-$ROLLOUT_HISTORY_DIR/val}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$SAVE_PATH/saved_models/rl_ckpt/latr/$EXPERIMENT_NAME}
LOG_PATH=${LOG_PATH:-$SAVE_PATH/qwen35_rl/log/latr/$EXPERIMENT_NAME}
SAVELOG_FILE_NAME=${SAVELOG_FILE_NAME:-train_${TIMESTAMP}.log}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1500}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4800}
ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4800}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4800}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-9}
VAL_FREQ=${VAL_FREQ:-6}
LATR_TREE_WORKERS=${LATR_TREE_WORKERS:-1}
LATR_TREE_MICRO_BATCH_SIZE=${LATR_TREE_MICRO_BATCH_SIZE:-1}
LATR_TREE_MAX_TOKENS=${LATR_TREE_MAX_TOKENS:-512}
LATR_WEIGHT_BUCKET_MB=${LATR_WEIGHT_BUCKET_MB:-128}

for name in NNODES NPUS_PER_NODE TRAINER_NNODES ROLLOUT_TP TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE ROLLOUT_N \
    LATR_TREE_WORKERS LATR_TREE_MICRO_BATCH_SIZE RAY_READY_TIMEOUT; do
    value=${!name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer: $value" >&2; exit 1; }
done
[[ "$NODE_RANK" =~ ^[0-9]+$ ]] && (( NODE_RANK < NNODES )) || { echo "Invalid NODE_RANK" >&2; exit 1; }
(( TRAINER_NNODES == NNODES )) || { echo "This script uses all NNODES for actor/rollout/ref." >&2; exit 1; }
(( NPUS_PER_NODE % ROLLOUT_TP == 0 )) || { echo "ROLLOUT_TP must divide NPUS_PER_NODE." >&2; exit 1; }
(( LATR_TREE_WORKERS <= NNODES * NPUS_PER_NODE && ROLLOUT_N >= 2 )) || { echo "Invalid LATR width/worker count." >&2; exit 1; }
(( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || { echo "PPO_MINI_BATCH_SIZE must divide TRAIN_BATCH_SIZE." >&2; exit 1; }
(( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % (NNODES * NPUS_PER_NODE) == 0 )) || {
    echo "PPO_MINI_BATCH_SIZE * ROLLOUT_N must be divisible by total training devices." >&2; exit 1;
}
(( MAX_MODEL_LEN >= MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH )) || { echo "MAX_MODEL_LEN is too small." >&2; exit 1; }

export NON_MEGATRON=true MULTI_STREAM_MEMORY_REUSE=2 TASK_QUEUE_ENABLE=1 ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000 CPU_AFFINITY_CONF=1 HCCL_BUFFSIZE=64
unset PYTORCH_NPU_ALLOC_CONF
export LD_LIBRARY_PATH="/usr/local/lib:/root/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HCCL_EXEC_TIMEOUT=7200 HCCL_CONNECT_TIMEOUT=7200 ACL_DEVICE_SYNC_TIMEOUT=7200 TORCH_DIST_TIMEOUT=3600
export HCCL_IF_BASE_PORT=64000 HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" TP_SOCKET_IFNAME="$SOCKET_IFNAME"
export HCCL_HOST_SOCKET_PORT_RANGE=auto HCCL_NPU_SOCKET_PORT_RANGE=auto
export ASCEND_GLOBAL_LOG_LEVEL=3 HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0
export RAY_gcs_server_rpc_server_thread_num=32 RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000 RAY_worker_register_timeout_seconds=1200
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 VLLM_RPC_TIMEOUT=600000
export RAY_HEARTBEAT_TIMEOUT_MS=60000 RAY_gcs_actor_failure_grace_period_ms=60000
export TOKENIZERS_PARALLELISM=false
export TMPDIR=${RAY_TMPDIR:-/cache/ray_tmp}
export NODE_RANK MASTER_ADDR MASTER_PORT
# Let Ray assign WORLD_SIZE/RANK to training workers.
unset WORLD_SIZE RANK LOCAL_RANK
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,$MASTER_ADDR${NODE_IP:+,$NODE_IP}${VC_WORKER_HOSTS:+,$VC_WORKER_HOSTS}"
export NO_PROXY="$no_proxy"

args=(
    trainer.use_v1=true trainer.v1.trainer_mode=sync ++trainer.v1.sync.parameter_sync_step=1
    algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
    "data.train_files=['$TRAIN_FILE']" "data.val_files=['$VAL_FILE_HIGH','$VAL_FILE_LOW']"
    data.image_key=images data.prompt_key=prompt data.reward_fn_key=data_source data.return_raw_chat=true
    "data.train_batch_size=$TRAIN_BATCH_SIZE" "data.gen_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH" "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=true data.truncation=error data.shuffle=true
    ++data.apply_chat_template_kwargs.enable_thinking=false
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=false actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.use_torch_compile=false
    "actor_rollout_ref.actor.optim.lr=$ACTOR_LR" "actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO"
    "actor_rollout_ref.actor.grad_clip=$GRAD_CLIP" "actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.use_dynamic_bsz=false
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=false actor_rollout_ref.actor.entropy_from_logits_with_chunking=false
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true actor_rollout_ref.actor.fsdp_config.offload_policy=true
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra']"
    actor_rollout_ref.ref.strategy=fsdp2 actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.ref.fsdp_config.offload_policy=true actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
    actor_rollout_ref.ref.use_torch_compile=false
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.dtype=bfloat16
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N" "actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE"
    "actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P" actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 actor_rollout_ref.rollout.enforce_eager=true
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.rollout.enable_chunked_prefill=false actor_rollout_ref.rollout.free_cache_engine=true
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.enable_prefix_caching=false
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.calculate_log_probs=false
    actor_rollout_ref.rollout.latr.enable=true
    actor_rollout_ref.rollout.latr.max_n_branch_per_token=2
    actor_rollout_ref.rollout.latr.prob_filter_abs_thres=0.25
    actor_rollout_ref.rollout.latr.prob_filter_rel_thres=0.15
    'actor_rollout_ref.rollout.latr.rollout_filter_steps=[20,30,50]'
    actor_rollout_ref.rollout.latr.rollout_filter_edit_dist_thres=0.4
    actor_rollout_ref.rollout.latr.mix_ratio_initial=1.0 actor_rollout_ref.rollout.latr.mix_ratio_decay=0.985
    "actor_rollout_ref.rollout.latr.force_return_step=$LATR_TREE_MAX_TOKENS"
    "actor_rollout_ref.rollout.latr.tree_micro_batch_size=$LATR_TREE_MICRO_BATCH_SIZE"
    "actor_rollout_ref.rollout.latr.tree_worker_count=$LATR_TREE_WORKERS"
    "actor_rollout_ref.rollout.latr.weight_bucket_mb=$LATR_WEIGHT_BUCKET_MB"
    actor_rollout_ref.rollout.latr.attn_implementation=sdpa
    reward.reward_manager.name=naive "reward.num_workers=$REWARD_NUM_WORKERS"
    "reward.custom_reward_function.path=$REWARD_FILE"
    trainer.balance_batch=true 'trainer.logger=[console,tensorboard]'
    "trainer.project_name=$PROJECT_NAME" "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.rollout_data_dir=$TRAIN_ROLLOUT_DIR" "trainer.validation_data_dir=$VAL_ROLLOUT_DIR"
    "trainer.n_gpus_per_node=$NPUS_PER_NODE" "trainer.nnodes=$TRAINER_NNODES" trainer.device=npu
    "trainer.default_local_dir=$DEFAULT_LOCAL_DIR" trainer.val_before_train=false
    "trainer.save_freq=$SAVE_FREQ" "trainer.test_freq=$VAL_FREQ" "trainer.total_epochs=$TOTAL_EPOCHS"
)
[[ -z "${TOTAL_STEPS:-}" ]] || args+=("trainer.total_training_steps=$TOTAL_STEPS")
args+=("$@")

# Print the exact argv without touching CANN, /cache, data, NPU, or Ray.
if [[ "$LATR_DRY_RUN" == 1 ]]; then
    printf '%s\n' -m verl.trainer.main_ppo "${args[@]}"
    exit 0
fi

for path in "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE_HIGH" "$VAL_FILE_LOW" "$REWARD_FILE"; do
    [[ -e "$path" ]] || { echo "Missing server path: $path" >&2; exit 1; }
done
# Retain the active dependency installation from grpo.sh; allow offline images
# with these dependencies already installed to skip it.
if [[ "${INSTALL_RUNTIME_DEPS:-1}" == 1 ]]; then
    "$PYTHON_BIN" -m pip install 'accelerate>=1.6.0' aiohttp aiofiles math-verify \
        -i "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
fi
"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import torch
import torch_npu
import verl
from verl.experimental.latr import hybrid
assert torch.npu.is_available(), "NPU is unavailable"
expected = Path(os.environ["PYTHONPATH"].split(":")[0]) / "verl"
assert Path(verl.__file__).resolve().parent == expected.resolve(), verl.__file__
print("Using adapted verl:", verl.__file__)
print("LATR coordinator:", hybrid.__file__)
PY
mkdir -p "$TMPDIR" "$LOG_PATH" "$TENSORBOARD_DIR" "$TRAIN_ROLLOUT_DIR" "$VAL_ROLLOUT_DIR"
[[ -n "$NODE_IP" ]] || { echo "Set NODE_IP or MA_CURRENT_IP on every node." >&2; exit 1; }

if (( NODE_RANK != 0 )); then
    deadline=$((SECONDS + RAY_READY_TIMEOUT))
    until ray start --node-ip-address="$NODE_IP" --address="$RAY_ADDRESS" --resources="{\"NPU\":$NPUS_PER_NODE}"; do
        (( SECONDS < deadline )) || { echo "Timed out joining Ray head $RAY_ADDRESS" >&2; exit 1; }
        sleep 10
    done
    sleep infinity
fi

ray start --head --node-ip-address="$NODE_IP" --port="$MASTER_PORT" \
    --dashboard-host=0.0.0.0 --dashboard-port="$DASHBOARD_PORT" --resources="{\"NPU\":$NPUS_PER_NODE}"
"$PYTHON_BIN" - "$NNODES" "$NPUS_PER_NODE" "$RAY_READY_TIMEOUT" <<'PY'
import sys
import time
import ray
nodes, devices, timeout = map(int, sys.argv[1:])
ray.init(address="auto")
deadline = time.monotonic() + timeout
while True:
    ready = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("NPU", 0) >= devices]
    print(f"Ray NPU nodes ready: {len(ready)}/{nodes}", flush=True)
    if len(ready) >= nodes:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError("Waiting for Ray NPU nodes timed out")
    time.sleep(10)
ray.shutdown()
PY

set +e
"$PYTHON_BIN" -m verl.trainer.main_ppo "${args[@]}" 2>&1 | tee -a "$LOG_PATH/$SAVELOG_FILE_NAME"
exit_codes=("${PIPESTATUS[@]}")
set -e
if (( exit_codes[0] != 0 )); then
    echo "LATR training failed: ${exit_codes[0]}" >&2
    exit "${exit_codes[0]}"
fi
exit "${exit_codes[1]}"
