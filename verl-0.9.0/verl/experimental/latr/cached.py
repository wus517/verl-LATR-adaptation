# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""One prefill per root, then batched single-token forwards in this process."""

import torch

from .core import Candidate, TreeState


def select_cache_rows(cache, indices: torch.Tensor, old_size: int):
    """Fork/prune *all* cache state, including Qwen3.5 conv/recurrent states.

    Modern Transformers Cache.reorder_cache supports duplicate indices, unlike
    the linear cache's batch_select_indices. Never crop/rollback recurrent state.
    """
    if indices.tolist() == list(range(old_size)):
        return cache
    if not callable(getattr(cache, "reorder_cache", None)):
        raise TypeError("LATR requires a Transformers Cache with reorder_cache, not legacy KV tuples")
    cache.reorder_cache(indices)
    return cache


@torch.inference_mode()
def generate_trees(model, tokenizer, config, requests, eos_ids, device):
    states = []
    for request in requests:
        state = TreeState(config, eos_ids, lambda token: tokenizer.decode([token]))
        state.start(request["n"], request["response_length"], request["global_step"])
        states.append(state)

    rows = [(i, branch.id) for i, state in enumerate(states) for branch in state.active]
    if not rows:
        return [state.result() for state in states]
    prompts = [requests[i]["prompt_ids"] for i, _ in rows]
    if any(not prompt for prompt in prompts):
        raise ValueError("LATR requires nonempty tokenized prompts")
    width = max(map(len, prompts))
    pad = tokenizer.pad_token_id
    pad = pad if pad is not None else next(iter(eos_ids), 0)
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, logits_to_keep=1)
    cache, logits = output.past_key_values, output.logits[:, -1, :]
    del output
    if cache is None or not callable(getattr(cache, "reorder_cache", None)):
        raise TypeError("Model must return a reorderable cache with use_cache=True")

    while rows:
        # Normalize before applying sampling filters; transfer only top-k to CPU.
        raw = logits.float()
        values, tokens = raw.topk(min(config.max_n_branch_per_token, raw.shape[-1]), dim=-1)
        logprobs = (values - raw.logsumexp(-1, keepdim=True)).tolist()
        token_lists = tokens.tolist()
        del raw, values, tokens, logits
        proposals = [{} for _ in states]
        source = {key: row for row, key in enumerate(rows)}
        for (group, branch), token_ids, probs in zip(rows, token_lists, logprobs, strict=True):
            proposals[group][branch] = [Candidate(t, p) for t, p in zip(token_ids, probs, strict=True)]
        for i, state in enumerate(states):
            if not state.done:
                state.advance(proposals[i])

        next_rows, parents, pending = [], [], []
        for group, state in enumerate(states):
            for branch in state.active:
                key = (group, branch.id)
                # New children inherit the cache immediately before the fork token.
                parent_key = key if key in source else (group, branch.parent)
                parents.append(source[parent_key])
                pending.append(branch.tokens[-1])
                next_rows.append(key)
        if not next_rows:
            break
        indices = torch.tensor(parents, dtype=torch.long, device=device)
        cache = select_cache_rows(cache, indices, len(rows))
        mask = mask.index_select(0, indices)
        next_positions = mask.sum(-1, keepdim=True)
        mask = torch.cat([mask, mask.new_ones((len(parents), 1))], dim=-1)
        output = model(
            input_ids=torch.tensor(pending, device=device).unsqueeze(-1),
            attention_mask=mask,
            position_ids=next_positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache, logits = output.past_key_values, output.logits[:, -1, :]
        del output
        rows = next_rows
    return [state.result() for state in states]
