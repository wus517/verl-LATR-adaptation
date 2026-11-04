# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Isolated HF inference actor. Do not apply verl's training forward patches here."""

import gc
import time
from dataclasses import asdict


class CachedTreeWorker:
    def __init__(self, model_config, latr_config, dtype, device_name):
        import torch

        from verl.experimental.latr.config import LATRConfig
        from verl.utils.device import get_torch_device
        from verl.utils.model import get_hf_auto_model_class

        self.device_api = get_torch_device()
        self.device_api.set_device(0)
        self.device = f"{device_name}:0"
        self.config = LATRConfig(**latr_config)
        self.tokenizer = model_config.tokenizer
        hf_config = model_config.hf_config
        text_config = hf_config.get_text_config()
        if getattr(text_config, "num_experts", 0) or getattr(text_config, "num_local_experts", 0):
            raise ValueError("Hybrid LATR initially supports dense models only")
        # The actor's already-resolved config includes user overrides. No optimizer,
        # FSDP wrapper, remove-padding forward, or training monkey patches are used.
        hf_config._attn_implementation = self.config.attn_implementation
        text_config._attn_implementation = self.config.attn_implementation
        if device_name == "npu" and text_config.model_type == "qwen3_5_text":
            # Optional CUDA/Triton FLA packages can be installed on an NPU host.
            # HF selects them by importability, so explicitly choose its native
            # torch implementations rather than accidentally invoking CUDA.
            from transformers.models.qwen3_5 import modeling_qwen3_5

            modeling_qwen3_5.FusedRMSNormGated = None
            modeling_qwen3_5.causal_conv1d_fn = None
            modeling_qwen3_5.causal_conv1d_update = None
            modeling_qwen3_5.chunk_gated_delta_rule = None
            modeling_qwen3_5.fused_recurrent_gated_delta_rule = None
        self.model = (
            get_hf_auto_model_class(hf_config)
            .from_pretrained(
                model_config.local_path,
                config=hf_config,
                torch_dtype=getattr(torch, dtype),
                trust_remote_code=model_config.trust_remote_code,
                attn_implementation=self.config.attn_implementation,
                device_map="cpu",
            )
            .eval()
        )
        self.parameters = dict(self.model.named_parameters(remove_duplicate=False))
        self.buffers = dict(self.model.named_buffers())
        self.version = None
        self.received = set()
        eos = getattr(model_config.generation_config, "eos_token_id", None)
        eos = eos if eos is not None else self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos]) - {None}

    def begin_weights(self):
        self.version = None
        self.received.clear()
        self.parameters = dict(self.model.named_parameters(remove_duplicate=False))
        self.buffers = dict(self.model.named_buffers())
        return {name: tensor.dtype for name, tensor in (self.parameters | self.buffers).items()}

    def load_bucket(self, weights):
        import torch

        with torch.no_grad():
            for name, tensor in weights:
                target = self.parameters.get(name)
                target = self.buffers.get(name) if target is None else target
                if target is None or target.shape != tensor.shape:
                    raise ValueError(f"LATR HF weight mismatch: {name} {tuple(tensor.shape)}")
                target.copy_(tensor)
                self.received.add(name)

    def finish_weights(self, version):
        # Tied parameters may be exported under one alias only.
        received_ptrs = {self.parameters[k].data_ptr() for k in self.received if k in self.parameters}
        missing = [k for k, p in self.parameters.items() if p.data_ptr() not in received_ptrs]
        if missing:
            raise ValueError(f"LATR tree model did not receive current actor parameters: {missing[:10]}")
        self.version = version

    def generate(self, requests, version):
        from verl.experimental.latr.cached import generate_trees

        if self.version != version:
            raise RuntimeError("LATR tree weights are stale")
        results, error = [], None
        started = time.perf_counter()
        try:
            self.model.to(self.device)
            for start in range(0, len(requests), self.config.tree_micro_batch_size):
                chunk = requests[start : start + self.config.tree_micro_batch_size]
                groups = generate_trees(self.model, self.tokenizer, self.config, chunk, self.eos_ids, self.device)
                for request, group in zip(chunk, groups, strict=True):
                    results.append(
                        dict(
                            index=request["index"],
                            prompt_ids=request["prompt_ids"],
                            trajectories=[asdict(t) for t in group.trajectories],
                            stats=group.stats,
                            version=version,
                            eos_ids=list(self.eos_ids),
                        )
                    )
        except Exception as exc:
            # Do not retain a traceback holding the decoder's device cache tensors
            # while waking vLLM. Propagate a fresh exception after cleanup.
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self.model.cpu()
            gc.collect()
            self.device_api.synchronize()
            self.device_api.empty_cache()
        if error:
            raise RuntimeError(f"LATR cached tree failed: {error}")
        elapsed = time.perf_counter() - started
        for result in results:
            result["stats"]["tree_phase_seconds"] = elapsed
        return results
