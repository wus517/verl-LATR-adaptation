# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""LATR settings shared by the algorithm and the rollout configuration."""

import math
from dataclasses import dataclass, field


@dataclass
class LATRConfig:
    enable: bool = False
    max_n_branch_per_token: int = 2
    prob_filter_abs_thres: float = 0.25
    prob_filter_rel_thres: float = 0.15
    rollout_filter_steps: list[int] = field(default_factory=lambda: [20, 30, 50])
    rollout_filter_edit_dist_thres: float | None = 0.4
    rollout_filter_suffix_match_thres: float | None = None
    rollout_filter_rouge_l_thres: float | None = None
    keep_math_symbols: bool = False
    mix_ratio_initial: float = 1.0
    mix_ratio_decay: float = 0.985
    mix_ratio_schedule: dict[int, float] = field(default_factory=dict)
    force_return_step: int = 512
    return_nb_thres_init: float = 1.0
    return_nb_thres_decay: float = 0.0
    tree_micro_batch_size: int = 1
    tree_worker_count: int = 1
    weight_bucket_mb: int = 128
    attn_implementation: str = "sdpa"

    def __post_init__(self):
        for name in (
            "max_n_branch_per_token",
            "force_return_step",
            "tree_micro_batch_size",
            "tree_worker_count",
            "weight_bucket_mb",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"LATR {name} must be an integer")
        if self.max_n_branch_per_token < 2:
            raise ValueError("LATR max_n_branch_per_token must be >= 2 (including the main candidate)")
        if not self.rollout_filter_steps or any(type(s) is not int or s < 1 for s in self.rollout_filter_steps):
            raise ValueError("LATR rollout_filter_steps must contain positive window lengths")
        if len(set(self.rollout_filter_steps)) != len(self.rollout_filter_steps):
            raise ValueError("LATR rollout_filter_steps must not contain duplicates")
        for name in (
            "prob_filter_abs_thres",
            "prob_filter_rel_thres",
            "rollout_filter_edit_dist_thres",
            "rollout_filter_suffix_match_thres",
            "rollout_filter_rouge_l_thres",
            "mix_ratio_initial",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"LATR {name} must be in [0, 1]")
        if not 0 < self.mix_ratio_decay <= 1:
            raise ValueError("LATR mix_ratio_decay must be in (0, 1]")
        if (
            self.force_return_step < -1
            or min(self.tree_micro_batch_size, self.tree_worker_count, self.weight_bucket_mb) < 1
        ):
            raise ValueError("LATR needs force_return_step >= -1 and positive batch/worker/bucket sizes")
        if self.attn_implementation not in ("sdpa", "eager"):
            raise ValueError("LATR tree attn_implementation must be sdpa or eager")
        if not 0 < self.return_nb_thres_init <= 1 or not 0 <= self.return_nb_thres_decay < self.return_nb_thres_init:
            raise ValueError("LATR saturation thresholds must satisfy 0 <= decay < initial <= 1")
        if self.mix_ratio_schedule:
            if 0 not in self.mix_ratio_schedule:
                raise ValueError("LATR mix_ratio_schedule must include step 0")
            if any(
                type(s) is not int or s < 0 or not math.isfinite(v) or not 0 <= v <= 1
                for s, v in self.mix_ratio_schedule.items()
            ):
                raise ValueError("LATR schedule steps must be nonnegative and ratios in [0, 1]")

    def mix_ratio(self, step: int) -> float:
        if step < 0:
            raise ValueError("LATR global step must be nonnegative")
        if self.mix_ratio_schedule:
            key = max(s for s in self.mix_ratio_schedule if s <= step)
            return self.mix_ratio_schedule[key]
        return self.mix_ratio_initial * self.mix_ratio_decay**step


def validate_training_config(config):
    """Reject unsupported combinations before model workers are initialized."""
    rollout = config.actor_rollout_ref.rollout
    latr = rollout.get("latr", {})
    if not latr.get("enable", False):
        return
    LATRConfig(**dict(latr))
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("LATR currently requires trainer.use_v1=True and trainer.v1.trainer_mode=sync")
    if rollout.name != "vllm" or rollout.mode != "async" or not rollout.free_cache_engine:
        raise ValueError("Hybrid LATR requires async vLLM with free_cache_engine=True")
    if rollout.n < 2 or rollout.calculate_log_probs:
        raise ValueError("LATR requires n>=2 and calculate_log_probs=False (actor recomputes logprobs)")
    actor, model = config.actor_rollout_ref.actor, config.actor_rollout_ref.model
    if actor.strategy != "fsdp2" or not actor.fsdp_config.offload_policy:
        raise ValueError("Hybrid LATR initially requires FSDP2 offload_policy=True")
    if not actor.fsdp_config.optimizer_offload:
        raise ValueError("Hybrid LATR requires actor.fsdp_config.optimizer_offload=True")
    if model.lora_rank or model.lora.get("rank", 0) or model.lora_adapter_path or actor.fsdp_config.qat.enable:
        raise ValueError("Hybrid LATR initially supports full dense weights without LoRA or QAT")
    if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        ref = config.actor_rollout_ref.ref
        if ref.strategy != "fsdp2" or not ref.fsdp_config.offload_policy:
            raise ValueError("Hybrid LATR requires reference FSDP2 offload_policy=True")
    if rollout.dtype != "bfloat16":
        raise ValueError("Hybrid LATR initially uses bfloat16 tree and continuation weights")
    if not config.actor_rollout_ref.hybrid_engine:
        raise ValueError("Hybrid LATR requires colocated hybrid_engine=True")
    if config.critic.get("enable", None) or config.algorithm.adv_estimator != "grpo":
        raise ValueError("Hybrid LATR initially supports GRPO without a critic")
    if config.distillation.enabled or config.reward.reward_model.enable:
        raise ValueError("Hybrid LATR initially supports function rewards without teacher/reward models")
    correction = config.algorithm.get("rollout_correction", {}) or {}
    if any(correction.get(k) for k in ("bypass_mode", "rollout_is", "rollout_rs")):
        raise ValueError("LATR does not expose a behavior-policy density for rollout correction or bypass mode")
    if rollout.agent.default_agent_loop != "single_turn_agent" or rollout.agent.agent_loop_config_path:
        raise ValueError("LATR supports only the built-in single_turn_agent")
    if rollout.agent.get("agent_loop_manager_class") or config.data.continuous_token.enable:
        raise ValueError("LATR requires the default V1 manager and continuous_token.enable=False")
    if rollout.multi_turn.enable or rollout.disaggregation.enabled or rollout.enable_rollout_routing_replay:
        raise ValueError("LATR does not support multi-turn, PD disaggregation, or router replay")
    if (rollout.mtp and rollout.mtp.enable) or model.mtp.enable:
        raise ValueError("Hybrid LATR initially requires MTP disabled")
    if rollout.ignore_eos or rollout.agent.custom_async_server.path:
        raise ValueError("LATR requires ignore_eos=False and the stock vLLM server")
    if config.algorithm.filter_groups.enable or config.trainer.v1.sampler.sync_refill_failed_groups:
        raise ValueError("Hybrid LATR initially requires group filtering and failed-group refill disabled")
    if config.trainer.v1.sync.get("parameter_sync_step", 1) != 1:
        raise ValueError("Hybrid LATR requires parameter_sync_step=1")
    if config.data.gen_batch_size not in (None, config.data.train_batch_size):
        raise ValueError("Hybrid LATR requires gen_batch_size=train_batch_size")
    if latr.get("tree_worker_count", 1) > config.trainer.nnodes * config.trainer.n_gpus_per_node:
        raise ValueError("LATR tree_worker_count exceeds training worker count")
    if config.skip.rollout_tq.enable or config.trainer.v1.sampler.custom_sampler.path:
        raise ValueError("Hybrid LATR initially requires rollout skip and custom replay samplers disabled")
    if not rollout.do_sample:
        raise ValueError("LATR training requires rollout.do_sample=True; per-prompt greedy is supported")
