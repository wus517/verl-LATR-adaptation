# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""LATR token backend using the existing, weight-synchronized vLLM service."""

from .core import Candidate


class VLLMTokenBackend:
    def __init__(self, client, prompt_ids, sampling_params, request_id, semaphore):
        self.client = client
        self.prompt_ids = list(prompt_ids)
        self.sampling_params = dict(sampling_params)
        self.request_id = request_id
        self.semaphore = semaphore
        self.version = None

    async def _request(self, prefix, params, key):
        async with self.semaphore:
            output = await self.client.generate(
                request_id=f"{self.request_id}-{key}",
                prompt_ids=self.prompt_ids + prefix,
                sampling_params=params,
            )
        if output.stop_reason == "aborted":
            raise RuntimeError("LATR generation was aborted")
        low = output.extra_fields.get("min_global_steps", output.extra_fields.get("global_steps"))
        high = output.extra_fields.get("max_global_steps", output.extra_fields.get("global_steps"))
        if low is None or low != high or (self.version is not None and self.version != low):
            raise RuntimeError("LATR group encountered missing or inconsistent model weight versions")
        self.version = low
        return output

    async def candidates(self, prefix, count, key):
        output = await self._request(
            prefix,
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 1,
                "repetition_penalty": 1.0,
                "logprobs": False,
                "_latr_top_logprobs": count,
            },
            key,
        )
        raw = output.extra_fields.get("latr_candidates")
        if not raw:
            raise RuntimeError("vLLM did not return raw LATR candidates; check the server extension and logprobs_mode")
        return [Candidate(int(item["token_id"]), float(item["logprob"])) for item in raw]

    async def complete(self, prefix, remaining, key):
        if remaining < 1:
            return []
        params = dict(self.sampling_params, max_tokens=remaining, logprobs=False)
        output = await self._request(prefix, params, key)
        if not output.token_ids:
            raise RuntimeError("LATR continuation returned no tokens")
        if len(output.token_ids) > remaining:
            raise RuntimeError("LATR continuation exceeded the remaining response budget")
        return list(output.token_ids)
