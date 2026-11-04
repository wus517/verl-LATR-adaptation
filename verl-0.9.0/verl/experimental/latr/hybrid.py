# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Full-batch phase boundary and ordinary vLLM continuation."""

import time
from dataclasses import asdict
from uuid import uuid4

from .config import LATRConfig
from .core import TreeState, gather_settled


def prepare_batch(trainer, batch):
    from verl.utils import tensordict_utils as tu

    started = time.perf_counter()
    requests = trainer.agent_loop_manager.prepare_latr_prompts(batch)
    # Never switch models for individual tree microbatches. Complete every tree
    # and release all HF caches before a single vLLM continuation is submitted.
    trainer.checkpoint_manager.sleep_replicas()
    version = trainer._latr_policy_version
    config = LATRConfig(**dict(trainer.config.actor_rollout_ref.rollout.latr))
    tree_requests, results = [], []
    for request in requests:
        state = TreeState(config, set(), lambda token: "")  # Planning only; no token comparisons here.
        state.start(request["n"], request["response_length"], request["global_step"])
        if not state.done:
            tree_requests.append(request)
        else:
            group = state.result()
            results.append(
                dict(
                    index=request["index"],
                    prompt_ids=request["prompt_ids"],
                    trajectories=[asdict(t) for t in group.trajectories],
                    stats=group.stats,
                    version=version,
                    eos_ids=[],
                )
            )
    if tree_requests:
        rank_results = trainer.actor_rollout_wg.generate_latr_trees(tree_requests, version)
        tree_results = [item for rank in rank_results for item in rank]
        results.extend(tree_results)
        if sorted(r["index"] for r in results) != sorted(r["index"] for r in requests):
            raise RuntimeError("LATR tree workers returned missing or duplicate prompt groups")
    trainer.checkpoint_manager.update_weights(version)
    by_index = {r["index"]: r for r in results}
    tu.assign_non_tensor_stack(batch, "latr_payload", [by_index.get(i) for i in range(len(batch))])
    trainer.timing_raw["latr_prepare"] = time.perf_counter() - started


async def finish_group(worker, payload, sampling_params, prompt, validate):
    from verl.experimental.agent_loop import AgentLoopOutput

    limit = worker.config.actor_rollout_ref.rollout.response_length
    version = payload["version"]
    eos = set(payload["eos_ids"])

    async def finish(index, trajectory):
        tokens = list(trajectory["tokens"])
        metrics = {}
        if len(tokens) < limit and not (tokens and tokens[-1] in eos):
            params = dict(sampling_params, max_tokens=limit - len(tokens))
            started = time.perf_counter()
            output = await worker.llm_client.generate(
                request_id=uuid4().hex,
                prompt_ids=payload["prompt_ids"] + tokens,
                sampling_params=params,
                priority=int(prompt.get("priority", 0)),
            )
            metrics["generate_sequences"] = time.perf_counter() - started
            for key in ("min_global_steps", "max_global_steps"):
                if output.extra_fields.get(key) != version:
                    raise RuntimeError(f"LATR tree/vLLM policy version mismatch: {key}={output.extra_fields.get(key)}")
            if output.stop_reason == "aborted":
                raise RuntimeError("LATR continuation was aborted")
            tokens.extend(output.token_ids)
        tokens = tokens[:limit]
        for i, token in enumerate(tokens):
            if token in eos:
                tokens = tokens[: i + 1]
                break
        output = AgentLoopOutput(
            prompt_ids=payload["prompt_ids"],
            response_ids=tokens,
            response_mask=[1] * len(tokens),
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "min_global_steps": version,
                "max_global_steps": version,
                "latr": dict(payload["stats"], **{k: v for k, v in trajectory.items() if k != "tokens"}),
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
        kwargs = {k: v for k, v in prompt.items() if k != "latr_payload"}
        await worker._agent_loop_postprocess(output, validate, session_id=index, **kwargs)

    await gather_settled([finish(i, t) for i, t in enumerate(payload["trajectories"])])
