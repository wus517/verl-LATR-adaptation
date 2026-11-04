# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Run on the target GPU/NPU host before training to verify raw top-token support."""

import argparse
import math


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.experimental.latr.vllm_contract import extract_candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 3?"}], tokenize=True, add_generation_prompt=True
    )
    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        logprobs_mode="raw_logprobs",
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    probes = []
    for temperature in (0.0, 0.7):
        params = SamplingParams(temperature=temperature, top_p=1.0, top_k=-1, max_tokens=1, logprobs=2)
        result = engine.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)[0].outputs[0]
        candidates = extract_candidates(result.logprobs, 2)
        if len(candidates) < 2:
            raise RuntimeError("Backend did not return two candidates")
        probes.append(candidates)
    for left, right in zip(*probes, strict=True):
        if left["token_id"] != right["token_id"] or not math.isclose(left["logprob"], right["logprob"], abs_tol=1e-4):
            raise RuntimeError(
                "Candidate probabilities changed with temperature; raw-logprob support needs investigation"
            )
    for candidate in probes[0]:
        prefix = prompt_ids + [candidate["token_id"]]
        result = engine.generate(
            [{"prompt_token_ids": prefix}], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False
        )[0].outputs[0]
        if not result.token_ids:
            raise RuntimeError("Candidate prefix continuation returned no tokens")
    print("PASS: top raw candidate probabilities and explicit-prefix continuation")
    print("This probe does not validate training, TransferQueue, or checkpoint weight synchronization.")


if __name__ == "__main__":
    main()
