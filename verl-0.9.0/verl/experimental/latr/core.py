# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Backtracking-free LATR over an asynchronous token backend.

Each instance generates exactly one prompt group. Stable branch IDs replace the
old tensor row genealogy; the backend owns prefix caching and model execution.
"""

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .config import LATRConfig


@dataclass(frozen=True)
class Candidate:
    token_id: int
    logprob: float


class TokenBackend(Protocol):
    async def candidates(self, prefix: list[int], count: int, key: str) -> list[Candidate]: ...

    async def complete(self, prefix: list[int], remaining: int, key: str) -> list[int]: ...


async def gather_settled(awaitables: Sequence[Awaitable]):
    """Drain siblings on errors/cancellation before a prompt can publish a terminal state."""
    tasks = [asyncio.ensure_future(item) for item in awaitables]
    try:
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    for output in outputs:
        if isinstance(output, BaseException):
            raise output
    return outputs


def edit_distance(a: list[int], b: list[int]) -> int:
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def rouge_l(a: list[int], b: list[int]) -> float:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if left == right else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(a) + len(b))


def suffix_match(a: list[int], b: list[int]) -> float:
    best = 0
    for left, right in ((a, b), (b, a)):
        for length in range(1, len(left) + 1):
            suffix = left[-length:]
            if any(right[i : i + length] == suffix for i in range(len(right) - length + 1)):
                best = max(best, length)
    return best / len(a)


@dataclass
class Branch:
    id: int
    parent: int | None
    birth_step: int
    tokens: list[int] = field(default_factory=list)


@dataclass
class Trajectory:
    tokens: list[int]
    kind: str
    prefix_length: int
    copied: bool = False


@dataclass
class GroupResult:
    trajectories: list[Trajectory]
    stats: dict


class LATRGenerator:
    def __init__(
        self,
        config: LATRConfig,
        backend: TokenBackend,
        eos_ids: set[int],
        decode_token: Callable[[int], str] | None = None,
    ):
        self.config = config
        self.backend = backend
        self.eos_ids = eos_ids
        self.decode_token = decode_token
        if config.keep_math_symbols and decode_token is None:
            raise ValueError("keep_math_symbols requires a token decoder")

    def _ended(self, tokens: list[int]) -> bool:
        return bool(tokens and tokens[-1] in self.eos_ids)

    def _trim(self, tokens: list[int], limit: int) -> list[int]:
        for i, token in enumerate(tokens[:limit]):
            if token in self.eos_ids:
                return tokens[: i + 1]
        return tokens[:limit]

    def _redundant(self, a: list[int], b: list[int]) -> bool:
        cfg = self.config
        return (
            (
                cfg.rollout_filter_edit_dist_thres is not None
                and edit_distance(a, b) / len(a) < cfg.rollout_filter_edit_dist_thres
            )
            or (
                cfg.rollout_filter_suffix_match_thres is not None
                and suffix_match(a, b) > cfg.rollout_filter_suffix_match_thres
            )
            or (cfg.rollout_filter_rouge_l_thres is not None and rouge_l(a, b) > cfg.rollout_filter_rouge_l_thres)
        )

    async def generate(self, n: int, max_new_tokens: int, global_step: int) -> GroupResult:
        if n < 1 or max_new_tokens < 1:
            raise ValueError("LATR group size and response length must be positive")
        cfg = self.config
        kt_count = round(n * cfg.mix_ratio(global_step))
        branches = [Branch(0, None, 0)] if kt_count else []
        next_id = 1
        stats = {
            "kt_count": kt_count,
            "std_count": n - kt_count,
            "tree_steps": 0,
            "created": 0,
            "suppressed": 0,
            "pruned": 0,
            "copied": 0,
            "checks": {str(r): 0 for r in cfg.rollout_filter_steps},
            "skipped_eos": 0,
            "skipped_math": 0,
            "exit_reason": "standard",
        }
        tree_limit = min(max_new_tokens, cfg.force_return_step) if cfg.force_return_step >= 0 else max_new_tokens
        if branches:
            stats["exit_reason"] = "length" if tree_limit == max_new_tokens else "force_return"
        for step in range(1, tree_limit + 1) if branches else ():
            active = [branch for branch in branches if not self._ended(branch.tokens)]
            if not active:
                stats["exit_reason"] = "eos"
                break
            proposals = await gather_settled(
                [
                    self.backend.candidates(
                        branch.tokens.copy(), cfg.max_n_branch_per_token, f"tree-{branch.id}-{step}"
                    )
                    for branch in active
                ]
            )
            alternatives = []
            for branch, candidates in zip(active, proposals, strict=True):
                candidates = sorted(candidates, key=lambda c: (-c.logprob, c.token_id))
                if not candidates or len({c.token_id for c in candidates}) != len(candidates):
                    raise ValueError("LATR backend must return nonempty, distinct candidates")
                if any(not math.isfinite(c.logprob) or c.logprob > 1e-5 for c in candidates):
                    raise ValueError("LATR backend returned invalid raw log probabilities")
                candidates = candidates[: cfg.max_n_branch_per_token]
                prefix = branch.tokens.copy()
                main = candidates[0]
                branch.tokens.append(main.token_id)
                for candidate in candidates[1:]:
                    if (
                        math.exp(candidate.logprob) > cfg.prob_filter_abs_thres
                        and math.exp(main.logprob) - math.exp(candidate.logprob) < cfg.prob_filter_rel_thres
                    ):
                        alternatives.append((candidate, branch, prefix))
            # Retain all original paths; allocate remaining width by candidate probability.
            alternatives.sort(key=lambda item: (-item[0].logprob, item[1].id, item[0].token_id))
            capacity = kt_count - len(branches)
            stats["suppressed"] += max(0, len(alternatives) - capacity)
            for candidate, parent, prefix in alternatives[:capacity]:
                branches.append(Branch(next_id, parent.id, step - 1, prefix + [candidate.token_id]))
                next_id += 1
                stats["created"] += 1

            by_id = {branch.id: branch for branch in branches}
            remove = set()
            for child in branches:
                age = step - child.birth_step
                if child.parent is None or age not in cfg.rollout_filter_steps:
                    continue
                parent = by_id[child.parent]
                a, b = child.tokens[-age:], parent.tokens[-age:]
                # Finished paths retain their real EOS; do not append fake tokens to age them.
                if any(token in self.eos_ids for token in a + b):
                    stats["skipped_eos"] += 1
                    continue
                if len(a) != age or len(b) != age:
                    raise RuntimeError("LATR parent and child windows lost alignment")
                if cfg.keep_math_symbols:
                    token_text = self.decode_token(b[0]).strip()
                    if token_text and all(c in "+-*/\\<>=()[]{}_^&%$0123456789" for c in token_text):
                        stats["skipped_math"] += 1
                        continue
                stats["checks"][str(age)] += 1
                if self._redundant(a, b):
                    remove.add(child.id)
            # Stable IDs keep comparisons aligned even when other branches are removed.
            for branch in branches:
                if branch.parent in remove:
                    remove.add(branch.id)
            stats["pruned"] += len(remove)
            branches = [branch for branch in branches if branch.id not in remove]
            stats["tree_steps"] = step
            if all(self._ended(branch.tokens) for branch in branches):
                stats["exit_reason"] = "eos"
                break
            threshold = cfg.return_nb_thres_init - cfg.return_nb_thres_decay * (step - 1) / max_new_tokens
            if len(branches) >= round(kt_count * threshold) and all(
                branch.parent is None or step - branch.birth_step >= max(cfg.rollout_filter_steps)
                for branch in branches
            ):
                stats["exit_reason"] = "saturated"
                break

        trajectories = [Trajectory(branch.tokens.copy(), "latr", len(branch.tokens)) for branch in branches]
        if branches:
            while len(trajectories) < kt_count:
                prefix = branches[0].tokens.copy()
                trajectories.append(Trajectory(prefix, "latr", len(prefix), copied=True))
                stats["copied"] += 1
        trajectories.extend(Trajectory([], "standard", 0) for _ in range(n - kt_count))

        async def finish(index, trajectory):
            if not self._ended(trajectory.tokens) and len(trajectory.tokens) < max_new_tokens:
                suffix = await self.backend.complete(
                    trajectory.tokens.copy(), max_new_tokens - len(trajectory.tokens), f"finish-{index}"
                )
                trajectory.tokens.extend(suffix)
            trajectory.tokens = self._trim(trajectory.tokens, max_new_tokens)
            return trajectory

        outputs = await gather_settled([finish(i, trajectory) for i, trajectory in enumerate(trajectories)])
        return GroupResult(outputs, stats)
