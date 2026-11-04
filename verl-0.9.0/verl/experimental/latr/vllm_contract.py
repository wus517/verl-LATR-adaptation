# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-free helpers for the vLLM candidate-request contract."""

import math


def pop_candidate_count(params, *, enabled, logprobs_mode):
    count = params.pop("_latr_top_logprobs", None)
    if count is None:
        return None
    if not enabled or logprobs_mode != "raw_logprobs":
        raise ValueError("LATR candidate requests require enabled LATR and raw_logprobs mode")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("LATR candidate count must be an integer >= 2")
    if params.get("max_tokens") != 1 or params.get("temperature") != 0:
        raise ValueError("LATR candidate requests must generate exactly one greedy token")
    if params.get("top_p", 1.0) != 1 or params.get("top_k", -1) != -1:
        raise ValueError("LATR candidate requests must not truncate the token distribution")
    return count


def extract_candidates(output_logprobs, count):
    if not output_logprobs or len(output_logprobs) != 1:
        raise RuntimeError("LATR expected raw log probabilities for exactly one generated token")
    candidates = [
        {"token_id": int(token), "logprob": float(value.logprob)}
        for token, value in output_logprobs[0].items()
        if math.isfinite(value.logprob)
    ]
    candidates.sort(key=lambda item: (-item["logprob"], item["token_id"]))
    if len(candidates) < count or any(item["logprob"] > 1e-5 for item in candidates):
        raise RuntimeError("vLLM returned invalid LATR raw log probabilities")
    return candidates[:count]
