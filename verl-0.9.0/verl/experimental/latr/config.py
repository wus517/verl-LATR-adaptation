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
    max_concurrent_requests: int = 64

    def __post_init__(self):
        for name in ("max_n_branch_per_token", "force_return_step", "max_concurrent_requests"):
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
        if self.force_return_step < -1 or self.max_concurrent_requests < 1:
            raise ValueError("LATR needs force_return_step >= -1 and max_concurrent_requests >= 1")
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
    """Fail before Ray startup for combinations the initial integration cannot execute."""
    rollout = config.actor_rollout_ref.rollout
    latr = rollout.get("latr", {})
    if not latr.get("enable", False):
        return
    LATRConfig(**dict(latr))
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "sync":
        raise ValueError("LATR currently requires trainer.use_v1=True and trainer.v1.trainer_mode=sync")
    if rollout.name != "vllm" or rollout.mode != "async" or rollout.logprobs_mode != "raw_logprobs":
        raise ValueError("LATR requires vllm async rollout with logprobs_mode=raw_logprobs")
    if rollout.n < 2:
        raise ValueError("LATR requires rollout.n >= 2")
    if rollout.calculate_log_probs:
        raise ValueError("LATR requires calculate_log_probs=False; the actor must recompute model log probabilities")
    correction = config.algorithm.get("rollout_correction", {}) or {}
    if any(correction.get(k) for k in ("bypass_mode", "rollout_is", "rollout_rs")):
        raise ValueError("LATR does not expose a behavior-policy density for rollout correction or bypass mode")
    if rollout.agent.default_agent_loop != "single_turn_agent" or rollout.agent.agent_loop_config_path:
        raise ValueError("LATR currently supports only the built-in single_turn_agent")
    if rollout.agent.get("agent_loop_manager_class") or config.data.continuous_token.enable:
        raise ValueError("LATR currently requires the default V1 manager and continuous_token.enable=False")
    if rollout.multi_turn.enable or rollout.disaggregation.enabled or rollout.enable_rollout_routing_replay:
        raise ValueError("LATR currently does not support multi-turn, PD disaggregation, or router replay")
    if rollout.mtp and rollout.mtp.enable_rollout:
        raise ValueError("LATR candidate requests do not support speculative rollout")
    if rollout.ignore_eos:
        raise ValueError("LATR requires ignore_eos=False")
    engine_kwargs = rollout.engine_kwargs.get("vllm", {}) or {}
    if engine_kwargs.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        raise ValueError("LATR engine_kwargs must not override raw_logprobs mode")
    if engine_kwargs.get("speculative_config"):
        raise ValueError("LATR candidate requests do not support engine-level speculative decoding")
    if rollout.agent.custom_async_server.path:
        raise ValueError("LATR requires the built-in vLLM server candidate extension")
