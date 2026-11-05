#!/bin/bash
set -x

# GRPO | Qwen3.5-35B | mm_k12_math | vLLM rollout | FSDP | Ascend 910B multi-node
# 多节点版本：纯 GRPO + 规则奖励（rule-only），无 GenRM
#
# 与 run_qwen3_5_grpo_910b_fsdp_reward_35b_image_test_zsg.sh 的差异：
#   - 删除所有 GRM_* 配置和 env 透传
#   - 删除 reward.reward_model.* 与 reward.custom_reward_function.*
#   - reward 流走 verl/utils/reward_score/__init__.py:default_compute_score，
#     按 parquet 里 data_source=="mm_k12_math" 分发到
#     mm_k12_math_penalty_07311944.compute_score（见 __init__.py:93-100）
#   - trainer 占用全部 NNODES（不再为 GenRM 预留 GRM_NNODES）
#
# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

echo "datetime: $(date +'%Y-%m-%d %H:%M:%S')"
echo $MA_CURRENT_HOST_IP
echo $MA_CURRENT_IP

################## 上海三方电信 代理配置 ##################
PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"

export LD_PRELOAD=/home/ma-user/Ascend/cann-8.5.2/lib64/libprofapi.so${LD_PRELOAD:+:$LD_PRELOAD}
echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-toolkit_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-910b-ops_8.5.2_linux-aarch64.run ${PROJECT_ROOT}
# cp ${cq_qwen_datasets}/tools/cann_packages/cann852/Ascend-cann-nnal_8.5.2_linux-aarch64.run ${PROJECT_ROOT}

# ls ${PROJECT_ROOT}
echo "***************************************** update cann *****************************************"
ls /home/ma-user/
ls /home/ma-user/Ascend/

# cd ${PROJECT_ROOT}
# chmod +777 ./Ascend-cann-*
# ./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-910b-ops_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-nnal_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/nnal/atb/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF
export HCCL_BUFFSIZE=64
export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

# echo "***************************************** Installing Pytorch *****************************************"
# python3 -m pip install --upgrade pip
# pip uninstall jsonnet -y || true
# pip install --upgrade pip

# cd ${PROJECT_ROOT}
# pip install TransferQueue
# pip install transferqueue-0.1.9.dev0-py3-none-any.whl
# pip install torch==2.9.0
# pip install torch_npu==2.9.0
# pip uninstall torch_c_dlpack_ext -y || true
# pip install triton_ascend==3.2.0
# pip install torch==2.9.0
# pip install torch_npu==2.9.0
python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"


# echo "*****************************************Start install transformers/vllm/vllm-ascend*****************************************"
# cd ${PROJECT_ROOT}
# pip install -r requirements-npu.txt
# pip install pre-commit pylatexenc wandb qwen_vl_utils hf_transfer mathruler

# pip install tokenizers==0.21.1 transformers==5.3.0 accelerate==1.2.0

# pip install vllm==0.18.0
# pip install vllm-ascend==0.18.0

# pip install -U hydra-core codetiming

# # ---- user-adjustable ----
# # MODEL_SIZE: 4B | 35B
# MODEL_SIZE=${MODEL_SIZE:-4B}

# # 模型路径 & 多节点配置（按模型规模预设）
# if [ "${MODEL_SIZE}" = "35B" ]; then
#     MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/opensource_multimodal_models_guiyang/Qwen/Qwen3.5-35B-A3B/Qwen/Qwen3___5-35B-A3B}
#     NNODES=${NNODES:-$MA_NUM_HOSTS}
#     EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_geo3k_grpo_910b_multi}
# else
#     MODEL_PATH=${MODEL_PATH:-/opt/huawei/dataset/wyb_qwen_small/Qwen3.5-4B}
#     NNODES=${NNODES:-2}
#     EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_4b_geo3k_grpo_910b_multi}
# fi

# <<< 修改：使用变量 BASE_PATH 指向 xyi 数据目录
# ================= 数据集 & 奖励函数路径（xyi） =================
# ${lnn_sh} 在环境中已定义；若没有，请手动改为绝对路径
BASE_PATH=${BASE_PATH:-${lhx_sh}}
SAVE_PATH=${SAVE_PATH:-${lhx_save}}
echo "BASE_PATH resolved to: ${BASE_PATH}"

XYI_DIR="${BASE_PATH}/xyi"
# TRAIN_FILE=${TRAIN_FILE:-${XYI_DIR}/extract_train.json}
# VAL_FILE=${VAL_FILE:-${XYI_DIR}/extract_val.json}
TRAIN_FILE=${TRAIN_FILE:-${XYI_DIR}/train_verl.json}
VAL_FILE_HIGH=${VAL_FILE_HIGH:-${XYI_DIR}/high_pass_val_verl.json}
VAL_FILE_LOW=${VAL_FILE_LOW:-${XYI_DIR}/low_pass_val_verl.json}
# 注意：你目录里实际文件名是 reward_xiaoyi.py，配置里写的是 reward_math_verifier.py
# 二选一即可，下面默认使用 reward_xiaoyi.py，若已改名为 reward_math_verifier.py 请自行切换
REWARD_FILE=${REWARD_FILE:-${XYI_DIR}/verify_choice_answer_new_v4.py}
# REWARD_FILE=${REWARD_FILE:-${XYI_DIR}/reward_math_verifier.py}

echo "TRAIN_FILE      = ${TRAIN_FILE}"
# echo "VAL_FILE    = ${VAL_FILE}"
echo "VAL_FILE_HIGH   = ${VAL_FILE_HIGH}"
echo "VAL_FILE_LOW    = ${VAL_FILE_LOW}"
echo "REWARD_FILE     = ${REWARD_FILE}"
pip install "accelerate>=1.6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import accelerate; print(accelerate.__version__)"
# ===============================================================
MODEL_PATH=${MODEL_PATH:-${BASE_PATH}/model/Qwen3_5_9B} #
#MODEL_PATH=${MODEL_PATH:-${edu_gy_zsg}/qwen35_rl/openllm_model/Qwen/Qwen3___5-9B}
# 基于以下ckpt，在学段遵从数据集上增训
# MODEL_PATH=${MODEL_PATH:-${cq_qwen_datasets}/saved_models/rl_ckpt/qwen35_35b_64c_grm_0820_1130/global_step_240/huggingface_merged}
NNODES=${NNODES:-$MA_NUM_HOSTS}
EXPERIMENT_NAME=${EXP_NAME}

# ================= TensorBoard 输出目录（带时间戳） =================
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TENSORBOARD_DIR="${SAVE_PATH}/tensorboard/grpo/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${TENSORBOARD_DIR}"
echo "TENSORBOARD_DIR = ${TENSORBOARD_DIR}"

# ========================== 保存rollout历史 =================================
ROLLOUT_HISTORY_DIR="${SAVE_PATH}/rollout_history/grpo/${EXP_NAME}_${TIMESTAMP}"

TRAIN_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/train"
VAL_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/val"

mkdir -p "${TRAIN_ROLLOUT_DIR}" "${VAL_ROLLOUT_DIR}"

echo "TRAIN_ROLLOUT_DIR = ${TRAIN_ROLLOUT_DIR}"
echo "VAL_ROLLOUT_DIR   = ${VAL_ROLLOUT_DIR}"
# ========================================================================

NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

# ================= 规则奖励（rule-only）配置 =================
# 不启用 GenRM，也不启动 reward_model router
# reward 走 verl/utils/reward_score/__init__.py:default_compute_score
# 按 data_source=="mm_k12_math" 分发到 mm_k12_math_penalty_07311944.compute_score
# 若需要切换规则函数（如换用 mm_k12_math_penalty_new），修改 __init__.py 里 mm_k12_math 分支即可
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
# ========================================================================

# GRPO 不需要区分 Student/Teacher，整节点资源全给 actor/rollout/ref
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}
# rule-only：trainer 占用全部 NNODES（不再为 GenRM 预留节点）
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}

# /opt/huawei/dataset/cq_qwen_datasets/exp_train_json/train_rl_datasets_hard_8437_middle_1563_with_q_0724.parquet
# /opt/huawei/dataset/cq_qwen_datasets/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_0724.parquet
# /opt/huawei/dataset/Geometry3K/geo3k/train.parquet
# /opt/huawei/dataset/Geometry3K/geo3k/test.parquet
# TRAIN_FILE=${TRAIN_FILE:-/opt/huawei/dataset/cq_qwen_datasets/exp_train_json/train_rl_datasets_hard_8437_middle_1563_with_q_0724.parquet}
# 0806和0814区别：只有extra_info的question字段有区别
# TRAIN_FILE=${TRAIN_FILE:-/opt/huawei/dataset/cq_qwen_datasets/exp_train_json/train_rl_datasets_hard_8437_middle_1563_with_q_no_prompt2_0814.parquet}
# 学段遵从训练集
# TRAIN_FILE=${TRAIN_FILE:-${edu_gy_zsg}/qwen35_rl/rl_data/stage_compliance/rl/merged_rubric_stage_jihe_2598_0824.parquet}
# lnn_sh
# 原始在下一行
# TRAIN_FILE=${TRAIN_FILE:-${cq_qwen_datasets}/rl_datasets/exp_rl_0910_397b/merged_all_rl_data_15k_0912.parquet}
# TRAIN_FILE=${TRAIN_FILE:-${lnn_sh}/xyi/train.json}
# TEST_FILE=${TEST_FILE:-${cq_qwen_datasets}/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_0724.parquet}
# 原始在下一行
# TEST_FILE=${TEST_FILE:-${cq_qwen_datasets}/exp_train_json/test_rl_datasets_hard_8437_middle_1563_math_no_prompt2_0806.parquet}

# 学段遵从测试集
# TEST_FILE=${TEST_FILE:-${edu_gy_zsg}/qwen35_rl/rl_data/stage_compliance/rl/merged_smoke_test_327_0824.parquet}

# verl_grpo_geo3k
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_math1w}

# TRAIN_BATCH_SIZE 和 PPO_MINI_BATCH_SIZE设置成卡数
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128} #
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32} #2
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1500}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4800}

ACTOR_LR=${ACTOR_LR:-1e-6} #1e-6
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7} #0.7 改小
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

# DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${cq_qwen_datasets}/saved_models/rl_ckpt/${EXP_NAME}}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${SAVE_PATH}/saved_models/rl_ckpt/grpo/${EXP_NAME}}

# ${cq_qwen_datasets}/saved_models/rl_ckpt
# LOG_PATH=${LOG_PATH:-${edu_gy_zsg}/qwen35_rl/log/${EXP_NAME}/}
LOG_PATH=${LOG_PATH:-${SAVE_PATH}/qwen35_rl/log/grpo/${EXP_NAME}/}
echo "LOG_PATH resolved to: ${LOG_PATH}"

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-9}
VAL_FREQ=${VAL_FREQ:-6}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-4800}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4800}
# MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}

### 启动分布式网络配置（多节点）
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=8
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"
# # 确保使用正确的网卡名称
# export GLOO_SOCKET_IFNAME=eth0
# export TP_SOCKET_IFNAME=eth0
# # 尝试强制 vLLM 使用特定的本地地址进行初始化
# export VLLM_RPC_TORCH_DIST_BACKEND=gloo

# pip install -U "ray"

# pip install numpy av ftfy bs4 beartype pandas==2.0.3 diffusers==0.30.3 accelerate==0.32.1 pandarallel datasets qwen_vl_utils imageio imageio-ffmpeg jsonargparse peft==0.7.1 "pydantic>=2.10.4" gpytorch==1.14

# pip install jsonnet==0.17.0
# pip install torchdata rich
# pip install tensordict

# ================= Ray 资源清理与环境就绪 =================
# echo "Cleaning up old Ray processes..."
# ray stop --force || true
# sleep 5
# rm -rf /cache/ray_tmp/* || true
# pkill -9 -f raylet || true
# pkill -9 -f plasma_store || true
# pkill -9 -f gcs_server || true
# echo "Waiting 20s for NPU to release resources..."
# sleep 20

# pip install flash-linear-attention==0.5.0

# cd ${PROJECT_ROOT}
# pip install "datasets==4.3.0"
# pip install numpy==1.26.4
# pip install triton-ascend==3.2.0
# pip install -U hydra-core

# pip uninstall torch_c_dlpack_ext -y || true
# pip install torch==2.9.0
# pip install torch_npu==2.9.0

# pip list

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export VLLM_RPC_TIMEOUT=600000

# transformers==5.3.0
# pip install mathruler pylatexenc

# 安装指定版本的transformer
# pip install tokenizers==0.21.1
# cd ${PROJECT_ROOT}/transformers
# pip install -e .
# cd ${PROJECT_ROOT}
pip install aiohttp aiofiles math-verify -i https://pypi.tuna.tsinghua.edu.cn/simple
pip list

export TENSORBOARD_DIR="${TENSORBOARD_DIR}"

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

echo $RAY_gcs_rpc_server_reconnect_timeout_s

# ================= 多节点：启动 Ray 集群 =================
if [ "${NODE_RANK}" = "0" ]; then
    # ---- 头节点：启动 Ray Head ----
    echo "Starting Ray HEAD node on ${MA_CURRENT_IP}..."
    ray start --head \
        --node-ip-address=${MA_CURRENT_IP} \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --resources='{"NPU": '${NPUS_PER_NODE}'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / NPUS_PER_NODE))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster ready: ${device_count} nodes"
            ray status

            mkdir -p ${LOG_PATH}
            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.use_kl_in_reward=False \
                data.train_files="['${TRAIN_FILE}']" \
                data.val_files="['${VAL_FILE_HIGH}', '${VAL_FILE_LOW}']" \
                data.image_key=images \
                data.prompt_key=prompt \
                data.reward_fn_key=data_source \
                data.return_raw_chat=True \
                data.train_batch_size=${TRAIN_BATCH_SIZE} \
                data.max_prompt_length=${MAX_PROMPT_LENGTH} \
                data.max_response_length=${MAX_RESPONSE_LENGTH} \
                data.filter_overlong_prompts=True \
                data.truncation='error' \
                data.shuffle=True \
                ++data.apply_chat_template_kwargs.enable_thinking=False \
                ++actor_rollout_ref.rollout.engine_kwargs.chat_template_kwargs.enable_thinking=False \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
                actor_rollout_ref.model.use_remove_padding=False \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.strategy=fsdp2 \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
                actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
                actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
                actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
                actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_dynamic_bsz=False \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.01 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.calculate_entropy=False \
                actor_rollout_ref.rollout.max_num_batched_tokens=4800 \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.offload_policy=False \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.actor.fsdp_config.grad_sync_every_micro_batch=True \
                actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']" \
                actor_rollout_ref.ref.strategy=fsdp2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
                actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
                actor_rollout_ref.rollout.n=${ROLLOUT_N} \
                actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
                actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.rollout.val_kwargs.do_sample=False \
                actor_rollout_ref.rollout.val_kwargs.temperature=0 \
                actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
                actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.rollout.enable_chunked_prefill=True \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
                actor_rollout_ref.rollout.enable_prefix_caching=False \
                actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
                reward.reward_manager.name=naive \
                reward.num_workers=${REWARD_NUM_WORKERS} \
                reward.custom_reward_function.path="${REWARD_FILE}" \
                trainer.balance_batch=True \
                trainer.logger='["console","tensorboard"]' \
                trainer.project_name=${PROJECT_NAME} \
                trainer.experiment_name=${EXPERIMENT_NAME} \
                trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
                trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" \
                trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE} \
                trainer.nnodes=${TRAINER_NNODES} \
                trainer.device=npu \
                trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
                trainer.val_before_train=False \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=${VAL_FREQ} \
                trainer.use_v1=true \
                trainer.v1.trainer_mode=sync \
                data.gen_batch_size=${TRAIN_BATCH_SIZE} \
                actor_rollout_ref.actor.loss_agg_mode=token-mean \
                actor_rollout_ref.ref.fsdp_config.offload_policy=true \
                actor_rollout_ref.rollout.mode=async \
                actor_rollout_ref.rollout.top_k=-1 \
                actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
                actor_rollout_ref.rollout.calculate_log_probs=false \
                actor_rollout_ref.rollout.latr.enable=true \
                actor_rollout_ref.rollout.latr.max_n_branch_per_token=2 \
                actor_rollout_ref.rollout.latr.prob_filter_abs_thres=0.25 \
                actor_rollout_ref.rollout.latr.prob_filter_rel_thres=0.15 \
                actor_rollout_ref.rollout.latr.rollout_filter_steps=[20,30,50] \
                actor_rollout_ref.rollout.latr.rollout_filter_edit_dist_thres=0.4 \
                actor_rollout_ref.rollout.latr.rollout_filter_suffix_match_thres=null \
                actor_rollout_ref.rollout.latr.rollout_filter_rouge_l_thres=null \
                actor_rollout_ref.rollout.latr.keep_math_symbols=false \
                actor_rollout_ref.rollout.latr.mix_ratio_initial=1.0 \
                actor_rollout_ref.rollout.latr.mix_ratio_decay=0.995 \
                actor_rollout_ref.rollout.latr.mix_ratio_schedule={} \
                actor_rollout_ref.rollout.latr.return_nb_thres_init=1.0 \
                actor_rollout_ref.rollout.latr.return_nb_thres_decay=0.0 \
                actor_rollout_ref.rollout.latr.force_return_step=${LATR_TREE_MAX_TOKENS:--1} \
                actor_rollout_ref.rollout.latr.max_concurrent_requests=${LATR_MAX_CONCURRENT_REQUESTS:-64} \
                trainer.total_epochs=${TOTAL_EPOCHS} 2>&1 | tee -a ${LOG_PATH}/${SAVELOG_FILE_NAME}
            python_exit_code=${PIPESTATUS[0]}
            if [ "${python_exit_code}" -ne 0 ]; then
                echo "Training failed with exit code ${python_exit_code}...."
                sleep 3h
                exit ${python_exit_code}
            fi
            break
        fi

        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
else
    # # ---- 工作节点：注册到 Ray Head ----
    # while true; do
    #     sleep 5

    #     # echo "Waiting for Ray Head (GCS) to be ready..."
    #     # while ! nc -z ${MASTER_ADDR} ${MASTER_PORT}; do
    #     #     sleep 30
    #     # done


    #     echo "Ray Head is up! Starting worker..."
    #     ray start \
    #         --node-ip-address=${MA_CURRENT_IP} \
    #         --address=${MASTER_ADDR}:${MASTER_PORT} \
    #         --resources='{"NPU": '${NPUS_PER_NODE}'}'

    #     if ray status; then
    #         echo "Worker connected to Ray cluster"
    #         sleep 600
    #         break
    #     fi

    #     echo "Failed to connect Ray head, retrying..."
    #     sleep 10
    # done

    echo "Ray Head is up! Starting worker..."

    ray start \
    --node-ip-address=${MA_CURRENT_IP} \
    --address=${MASTER_ADDR}:${MASTER_PORT} \
    --resources='{"NPU": '${NPUS_PER_NODE}'}'

    sleep infinity
fi