# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Translate one LATR prompt group into ordinary V1 session outputs."""

import asyncio
import time

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from .backend import VLLMTokenBackend
from .core import LATRGenerator, gather_settled


async def run_latr_group(worker, prompt, sampling_params, trajectory, n):
    if prompt.get("agent_name", "single_turn_agent") != "single_turn_agent":
        raise ValueError("LATR supports only single_turn_agent prompts")
    messages = list(prompt["raw_prompt"])
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("LATR currently accepts single-turn text prompts only")
    formatter = SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(worker.config),
        server_manager=worker.llm_client,
        tokenizer=worker.tokenizer,
        processor=worker.processor,
        dataset_cls=worker.dataset_cls,
        data_config=DictConfigWrap(worker.config.data),
    )
    prompt_ids = await formatter.apply_chat_template(messages)
    eos_ids = set()
    for source in (
        worker.tokenizer,
        worker.model_config.hf_config,
        getattr(worker.model_config.hf_config, "text_config", None),
        worker.model_config.generation_config,
    ):
        ids = getattr(source, "eos_token_id", None)
        if ids is not None:
            eos_ids.update(ids if isinstance(ids, list | tuple | set) else [ids])
    if not eos_ids:
        raise ValueError("LATR requires a tokenizer/model EOS token ID")
    if not hasattr(worker, "_latr_request_semaphore"):
        worker._latr_request_semaphore = asyncio.Semaphore(worker.rollout_config.latr.max_concurrent_requests)
    backend = VLLMTokenBackend(
        worker.llm_client, prompt_ids, sampling_params, f"latr-{prompt['uid']}", worker._latr_request_semaphore
    )
    generator = LATRGenerator(
        worker.rollout_config.latr, backend, eos_ids, lambda token: worker.tokenizer.decode([token])
    )
    start = time.perf_counter()
    result = await generator.generate(n, worker.rollout_config.response_length, int(trajectory["step"]))
    elapsed = time.perf_counter() - start

    async def publish(session_id, path):
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=path.tokens,
            response_mask=[1] * len(path.tokens),
            multi_modal_data={},
            num_turns=2,
            metrics={"generate_sequences": elapsed},
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "min_global_steps": backend.version,
                "max_global_steps": backend.version,
                "latr": dict(result.stats, kind=path.kind, copied=path.copied, prefix_length=path.prefix_length),
            },
        )
        await worker._agent_loop_postprocess(output, False, session_id=session_id, **prompt)

    await gather_settled([publish(i, path) for i, path in enumerate(result.trajectories)])
