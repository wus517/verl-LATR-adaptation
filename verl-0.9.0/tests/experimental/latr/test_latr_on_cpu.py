# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic LATR behavior tests; runnable without torch, Ray or vLLM.

Framework boundary tests execute the real methods with fake external services.
They do not exercise device kernels, distributed execution or real TransferQueue.
"""

import ast
import asyncio
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "verl/experimental/latr"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Isolate pure algorithm imports from verl/__init__.py's torch dependency.
namespace = types.ModuleType("_latr_under_test")
namespace.__path__ = [str(PACKAGE)]
sys.modules[namespace.__name__] = namespace
config_module = load_file("_latr_under_test.config", PACKAGE / "config.py")
core = load_file("_latr_under_test.core", PACKAGE / "core.py")
backend_module = load_file("_latr_under_test.backend", PACKAGE / "backend.py")
contract = load_file("_latr_under_test.vllm_contract", PACKAGE / "vllm_contract.py")
reward = load_file("_latr_reward_under_test", ROOT / "examples/latr/reward.py")
data = load_file("_latr_data_under_test", ROOT / "examples/latr/prepare_data.py")
LATRConfig = config_module.LATRConfig
Candidate = core.Candidate


def candidates(*probabilities):
    return [Candidate(token, math.log(probability)) for token, probability in probabilities]


class ScriptedBackend:
    def __init__(self, propose=None, complete=None):
        self.propose = propose or (lambda prefix: candidates((1, 0.5), (2, 0.4)))
        self.continue_path = complete or (lambda prefix, remaining: [9] * remaining)
        self.probes = []
        self.completions = []

    async def candidates(self, prefix, count, key):
        self.probes.append((prefix.copy(), count, key))
        return self.propose(prefix)

    async def complete(self, prefix, remaining, key):
        self.completions.append((prefix.copy(), remaining, key))
        return self.continue_path(prefix, remaining)


class ConfigTests(unittest.TestCase):
    def test_schedule_and_resume_step(self):
        cfg = LATRConfig(mix_ratio_schedule={0: 1.0, 3: 0.5, 7: 0.0})
        self.assertEqual([cfg.mix_ratio(s) for s in [0, 2, 3, 6, 7, 100]], [1, 1, 0.5, 0.5, 0, 0])
        self.assertAlmostEqual(LATRConfig().mix_ratio(10), 0.985**10)

    def test_reject_invalid_config(self):
        for overrides in (
            {"max_n_branch_per_token": 1},
            {"max_n_branch_per_token": 2.5},
            {"rollout_filter_steps": []},
            {"rollout_filter_steps": [1, 1]},
            {"rollout_filter_steps": [1.5]},
            {"mix_ratio_schedule": {2: 0.5}},
            {"mix_ratio_schedule": {0: 1, 1.5: 0}},
            {"prob_filter_abs_thres": float("nan")},
            {"force_return_step": -2},
            {"max_concurrent_requests": 0},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                LATRConfig(**overrides)


class CoreTests(unittest.IsolatedAsyncioTestCase):
    async def run_tree(self, backend=None, n=4, length=5, step=0, **options):
        cfg = LATRConfig(**dict({"force_return_step": 1}, **options))
        backend = backend or ScriptedBackend()
        result = await core.LATRGenerator(cfg, backend, {99}).generate(n, length, step)
        return result, backend

    async def test_raw_thresholds_do_not_renormalize_top_two(self):
        # A conditional top-two renormalization would incorrectly allow this branch.
        backend = ScriptedBackend(lambda _: candidates((1, 0.24), (2, 0.23)))
        result, _ = await self.run_tree(backend, n=2)
        self.assertEqual(result.stats["created"], 0)
        self.assertEqual(result.stats["copied"], 1)

    async def test_strict_probability_thresholds(self):
        for pair, expected in [((0.45, 0.25), 0), ((0.5, 0.3), 0), ((0.45, 0.35), 1)]:
            with self.subTest(pair=pair):
                result, _ = await self.run_tree(
                    ScriptedBackend(lambda _, p=pair: candidates((1, p[0]), (2, p[1]))), n=2
                )
                self.assertEqual(result.stats["created"], expected)

    async def test_capacity_prefers_global_candidate_probability(self):
        def propose(prefix):
            if not prefix:
                return candidates((1, 0.5), (2, 0.4))
            return candidates((3, 0.5), (4, 0.36 if prefix[0] == 1 else 0.42))

        result, _ = await self.run_tree(ScriptedBackend(propose), n=3, force_return_step=2)
        self.assertEqual([p.tokens[:2] for p in result.trajectories], [[1, 3], [2, 3], [2, 4]])
        self.assertEqual(result.stats["suppressed"], 1)

    async def test_window_counts_branch_token_and_prunes_descendants(self):
        def propose(prefix):
            if not prefix:
                return candidates((1, 0.5), (2, 0.4))
            if prefix == [2]:
                return candidates((3, 0.5), (4, 0.4))
            return candidates((3, 0.9), (8, 0.01))

        result, backend = await self.run_tree(
            ScriptedBackend(propose),
            n=4,
            length=6,
            force_return_step=3,
            rollout_filter_steps=[3],
            rollout_filter_edit_dist_thres=0.4,
        )
        # Child [2,3,3] differs from parent [1,3,3] in 1/3 positions.
        # Its child [2,4,3] must disappear even though its own window is not due yet.
        self.assertEqual(result.stats["checks"], {"3": 1})
        self.assertEqual(result.stats["pruned"], 2)
        self.assertEqual(result.stats["copied"], 3)
        self.assertTrue(all(prefix == [1, 3, 3] for prefix, _, _ in backend.completions))

    async def test_all_windows_fire_once_at_expected_age(self):
        backend = ScriptedBackend(
            lambda p: candidates((1, 0.5), (2, 0.4)) if not p else candidates((3, 0.9), (4, 0.01))
        )
        result, _ = await self.run_tree(
            backend, n=2, length=60, force_return_step=-1, rollout_filter_edit_dist_thres=0.0
        )
        self.assertEqual(result.stats["checks"], {"20": 1, "30": 1, "50": 1})
        self.assertEqual(result.stats["tree_steps"], 50)
        self.assertEqual(result.stats["exit_reason"], "saturated")

    async def test_math_exemption_keeps_correct_pair_alignment(self):
        cfg = LATRConfig(
            force_return_step=1,
            max_n_branch_per_token=3,
            rollout_filter_steps=[1],
            rollout_filter_edit_dist_thres=None,
            rollout_filter_rouge_l_thres=0.5,
            keep_math_symbols=True,
        )
        # Two sibling comparisons share the mathematical parent token.
        backend = ScriptedBackend(lambda _: candidates((1, 0.36), (2, 0.33), (3, 0.3)))
        result = await core.LATRGenerator(cfg, backend, {99}, lambda token: "+" if token == 1 else "word").generate(
            3, 2, 0
        )
        self.assertEqual(result.stats["skipped_math"], 2)
        self.assertEqual([p.tokens[0] for p in result.trajectories], [1, 2, 3])

    async def test_eta_zero_one_and_mixed(self):
        for eta, expected in [(0, 0), (0.5, 2), (1, 4)]:
            with self.subTest(eta=eta):
                result, _ = await self.run_tree(mix_ratio_initial=eta)
                self.assertEqual(len(result.trajectories), 4)
                self.assertEqual(sum(p.kind == "latr" for p in result.trajectories), expected)
                self.assertTrue(all(len(p.tokens) == 5 for p in result.trajectories))

    async def test_zero_tree_budget_completes_full_response(self):
        result, backend = await self.run_tree(force_return_step=0, n=3)
        self.assertFalse(backend.probes)
        self.assertEqual(len(backend.completions), 3)
        self.assertTrue(all(p.tokens == [9] * 5 and p.prefix_length == 0 for p in result.trajectories))

    async def test_eos_paths_never_continue_and_completion_trims(self):
        result, backend = await self.run_tree(ScriptedBackend(lambda _: candidates((99, 0.9), (1, 0.01))), n=3)
        self.assertFalse(backend.completions)
        self.assertTrue(all(p.tokens == [99] for p in result.trajectories))
        result, _ = await self.run_tree(
            ScriptedBackend(complete=lambda p, r: [3, 99, 5, 5][:r]), mix_ratio_initial=0, n=1
        )
        self.assertEqual(result.trajectories[0].tokens, [3, 99])

    async def test_response_contains_prefix_and_suffix(self):
        result, backend = await self.run_tree(n=2, length=3)
        self.assertEqual([p.tokens for p in result.trajectories], [[1, 9, 9], [2, 9, 9]])
        self.assertEqual([r for _, r, _ in backend.completions], [2, 2])
        self.assertEqual([p.prefix_length for p in result.trajectories], [1, 1])

    async def test_invalid_backend_candidates_fail(self):
        for raw in [[], [Candidate(1, 0.1)], [Candidate(1, float("nan"))], candidates((1, 0.5), (1, 0.4))]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                await self.run_tree(ScriptedBackend(lambda _, raw=raw: raw))

    async def test_failure_waits_for_siblings(self):
        done = []

        async def fail():
            raise RuntimeError("failed")

        async def sibling():
            await asyncio.sleep(0)
            done.append("settled")

        with self.assertRaisesRegex(RuntimeError, "failed"):
            await core.gather_settled([fail(), sibling()])
        self.assertEqual(done, ["settled"])

    async def test_cancellation_drains_children(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def child():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        task = asyncio.create_task(core.gather_settled([child()]))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())


class ContractTests(unittest.TestCase):
    def test_internal_count_preserved_and_not_booleanized(self):
        params = dict(_latr_top_logprobs=2, max_tokens=1, temperature=0, top_p=1, top_k=-1)
        self.assertEqual(contract.pop_candidate_count(params, enabled=True, logprobs_mode="raw_logprobs"), 2)
        self.assertNotIn("_latr_top_logprobs", params)
        self.assertIsNone(contract.pop_candidate_count({}, enabled=False, logprobs_mode="processed_logprobs"))

    def test_reject_invalid_candidate_requests(self):
        for update in [{"_latr_top_logprobs": True}, {"temperature": 1}, {"max_tokens": 2}, {"top_p": 0.9}]:
            params = dict(_latr_top_logprobs=2, max_tokens=1, temperature=0)
            params.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                contract.pop_candidate_count(params, enabled=True, logprobs_mode="raw_logprobs")
        with self.assertRaises(ValueError):
            contract.pop_candidate_count({"_latr_top_logprobs": 2}, enabled=True, logprobs_mode="processed_logprobs")

    def test_extraction_sorts_raw_values_and_requires_count(self):
        raw = [{2: NS(logprob=-2.0), 1: NS(logprob=-0.5), 3: NS(logprob=-3.0)}]
        self.assertEqual(
            contract.extract_candidates(raw, 2),
            [
                {"token_id": 1, "logprob": -0.5},
                {"token_id": 2, "logprob": -2.0},
            ],
        )
        with self.assertRaises(RuntimeError):
            contract.extract_candidates([{1: NS(logprob=-0.5)}], 2)


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def make_backend(self, versions=(7,), stop_reason="completed", tokens=None):
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            version = versions[min(len(calls) - 1, len(versions) - 1)]
            return NS(
                token_ids=[8] if tokens is None else tokens,
                stop_reason=stop_reason,
                extra_fields={
                    "global_steps": version,
                    "latr_candidates": [
                        {"token_id": 1, "logprob": -0.5},
                        {"token_id": 2, "logprob": -1.0},
                    ],
                },
            )

        backend = backend_module.VLLMTokenBackend(
            NS(generate=generate), [100, 101], {"temperature": 0.7}, "uid-a", asyncio.Semaphore(1)
        )
        return backend, calls

    async def test_explicit_prefix_candidate_parameters_and_completion_budget(self):
        backend, calls = self.make_backend()
        await backend.candidates([1], 2, "tree-0")
        await backend.complete([1, 2], 3, "finish-0")
        self.assertEqual(calls[0]["prompt_ids"], [100, 101, 1])
        self.assertEqual(calls[0]["sampling_params"]["_latr_top_logprobs"], 2)
        self.assertEqual(calls[0]["sampling_params"]["temperature"], 0)
        self.assertEqual(calls[1]["sampling_params"]["temperature"], 0.7)
        self.assertEqual(calls[1]["sampling_params"]["max_tokens"], 3)
        self.assertNotEqual(calls[0]["request_id"], calls[1]["request_id"])
        self.assertEqual(backend.version, 7)

    async def test_weight_version_cannot_change_within_group(self):
        backend, _ = self.make_backend(versions=(7, 8))
        await backend.candidates([], 2, "tree")
        with self.assertRaisesRegex(RuntimeError, "weight versions"):
            await backend.complete([1], 1, "finish")

    async def test_missing_version_abort_empty_and_over_budget_fail(self):
        for options in [{"versions": (None,)}, {"stop_reason": "aborted"}, {"tokens": []}, {"tokens": [1, 2]}]:
            backend, _ = self.make_backend(**options)
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                await backend.complete([], 1, "finish")


def extract_methods(path, class_name, names, globals_dict):
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            body.extend(item for item in node.body if isinstance(item, ast.AsyncFunctionDef) and item.name in names)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names:
            body.append(node)
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + body,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), globals_dict)
    return globals_dict


class FrameworkBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def agent_module(self):
        # Replace only external imports; all LATR bridge code executes unchanged.
        class FakeOutput(NS):
            pass

        class FakeFormatter:
            def __init__(self, **kwargs):
                pass

            async def apply_chat_template(self, messages):
                return [100, 101]

        with patch.dict(
            sys.modules,
            {
                "verl.experimental.agent_loop.agent_loop": NS(AgentLoopOutput=FakeOutput, DictConfigWrap=lambda x: x),
                "verl.experimental.agent_loop.single_turn_agent_loop": NS(SingleTurnAgentLoop=FakeFormatter),
            },
        ):
            return load_file("_latr_under_test.agent_loop", PACKAGE / "agent_loop.py")

    async def make_worker(self):
        agent = self.agent_module()
        backend, calls = BackendTests().make_backend(tokens=[99])
        published = []

        async def postprocess(output, validate, **kwargs):
            published.append((output, validate, kwargs))

        worker = NS(
            config=NS(data={}),
            rollout_config=NS(latr=LATRConfig(force_return_step=1), response_length=5),
            tokenizer=NS(eos_token_id=99, decode=lambda ids: "word"),
            processor=object(),
            dataset_cls=object,
            model_config=NS(hf_config=NS(), generation_config=NS()),
            llm_client=backend.client,
            _agent_loop_postprocess=postprocess,
        )
        return agent, worker, published, calls

    async def test_group_publishes_exact_n_complete_responses_and_masks(self):
        agent, worker, published, _ = await self.make_worker()
        await agent.run_latr_group(worker, {"uid": "A", "raw_prompt": [{"content": "question"}]}, {}, {"step": 0}, 3)
        self.assertEqual(len(published), 3)
        self.assertEqual(sorted(p[2]["session_id"] for p in published), [0, 1, 2])
        for output, validate, kwargs in published:
            self.assertEqual(output.prompt_ids, [100, 101])
            self.assertIn(output.response_ids[0], [1, 2])
            self.assertEqual(output.response_ids[-1], 99)
            self.assertEqual(output.response_mask, [1] * len(output.response_ids))
            self.assertEqual(output.extra_fields["min_global_steps"], 7)
            self.assertFalse(validate)
            self.assertEqual(kwargs["uid"], "A")

    async def test_identical_prompts_keep_distinct_uid_and_request_ids(self):
        agent, worker, published, calls = await self.make_worker()
        await asyncio.gather(
            *[
                agent.run_latr_group(worker, {"uid": uid, "raw_prompt": [{"content": "same"}]}, {}, {"step": 0}, 2)
                for uid in ("A", "B")
            ]
        )
        self.assertEqual({p[2]["uid"] for p in published}, {"A", "B"})
        self.assertEqual(len({c["request_id"] for c in calls}), len(calls))

    async def test_group_publication_failure_drains_sibling_writes(self):
        agent, worker, _, _ = await self.make_worker()
        settled = []

        async def postprocess(output, validate, session_id, **kwargs):
            if session_id == 0:
                raise RuntimeError("write failed")
            await asyncio.sleep(0)
            settled.append(session_id)

        worker._agent_loop_postprocess = postprocess
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            await agent.run_latr_group(
                worker, {"uid": "A", "raw_prompt": [{"content": "question"}]}, {}, {"step": 0}, 3
            )
        self.assertEqual(sorted(settled), [1, 2])

    async def test_tq_failure_and_cancellation_publish_terminal_state_last(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                events = []
                entered = asyncio.Event()

                async def put(events=events, **kwargs):
                    events.append(kwargs["tag"]["status"])

                async def tree(*args, events=events, entered=entered, cancel=cancel):
                    try:
                        entered.set()
                        if cancel:
                            await asyncio.Event().wait()
                        else:
                            raise RuntimeError("group failed")
                    finally:
                        events.append("tree-settled")

                env = extract_methods(
                    ROOT / "verl/trainer/ppo/v1/agent_loop_tq.py",
                    "AgentLoopWorkerTQ",
                    {"_run_prompt", "_settle_session_tasks", "apply_greedy_sampling_params"},
                    {"asyncio": asyncio, "tq": NS(async_kv_put=put), "logger": NS(error=lambda *a, **k: None)},
                )
                worker = NS(
                    config=NS(actor_rollout_ref=NS(rollout=NS(n=3))),
                    rollout_config=NS(latr=LATRConfig(enable=True)),
                )
                with patch.dict(sys.modules, {"verl.experimental.latr.agent_loop": NS(run_latr_group=tree)}):
                    task = asyncio.create_task(env["_run_prompt"](worker, {"uid": "A"}, {}, {"validate": False}))
                    await entered.wait()
                    if cancel:
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        await task
                self.assertEqual(events, ["running", "tree-settled", "failure"])

    async def test_tq_routes_one_tree_and_validation_or_greedy_to_independent_sessions(self):
        for validate, do_sample, expected_tree in [(False, True, True), (True, True, False), (False, False, False)]:
            with self.subTest(validate=validate, do_sample=do_sample):
                events = []

                async def put(events=events, **kwargs):
                    events.append(kwargs["tag"]["status"])

                async def tree(*args, events=events):
                    events.append("tree")

                async def standard(params, do_sample=do_sample, events=events, **kwargs):
                    if not do_sample:
                        self.assertEqual(params["temperature"], 0)
                    events.append(f"session-{kwargs['session_id']}")

                env = extract_methods(
                    ROOT / "verl/trainer/ppo/v1/agent_loop_tq.py",
                    "AgentLoopWorkerTQ",
                    {"_run_prompt", "_settle_session_tasks", "apply_greedy_sampling_params"},
                    {"asyncio": asyncio, "tq": NS(async_kv_put=put), "logger": logging.getLogger(__name__)},
                )
                worker = NS(
                    config=NS(actor_rollout_ref=NS(rollout=NS(n=3, val_kwargs=NS(n=2)))),
                    rollout_config=NS(latr=LATRConfig(enable=True)),
                    _run_agent_loop=standard,
                )
                with patch.dict(sys.modules, {"verl.experimental.latr.agent_loop": NS(run_latr_group=tree)}):
                    await env["_run_prompt"](
                        worker, {"uid": "A", "__do_sample__": do_sample}, {"temperature": 0.7}, {"validate": validate}
                    )
                self.assertEqual(events[0], "running")
                self.assertEqual(events[-1], "finished")
                self.assertEqual(
                    events[1:-1], ["tree"] if expected_tree else [f"session-{i}" for i in range(2 if validate else 3)]
                )

    async def test_actual_vllm_generate_preserves_candidate_count(self):
        seen = []

        async def generate(**kwargs):
            seen.append(kwargs)
            yield NS(
                outputs=[
                    NS(token_ids=[1], logprobs=[{1: NS(logprob=-0.5), 2: NS(logprob=-1.0)}], finish_reason="length")
                ]
            )

        def sampling(**kwargs):
            return NS(prompt_logprobs=None, **kwargs)

        env = extract_methods(
            ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
            "vLLMHttpServer",
            {"generate"},
            {
                "normalize_token_ids": list,
                "SamplingParams": sampling,
                "TokenOutput": NS,
                "qwen2_5_vl_dedup_image_tokens": lambda ids, processor: ids,
                "TokensPrompt": lambda **kwargs: kwargs,
                "extract_prompt_logprobs": lambda **kwargs: None,
                "RLInsightLogger": NS(trace_state=lambda *args, **kwargs: nullcontext()),
            },
        )

        class Mapping(NS):
            def get(self, key, default=None):
                return getattr(self, key, default)

        server = NS(
            _disaggregation_role=None,
            config=Mapping(
                latr=LATRConfig(enable=True),
                logprobs_mode="raw_logprobs",
                max_model_len=16,
                response_length=5,
                prompt_length=5,
                full_determinism=False,
                enable_rollout_routing_replay=False,
                mtp=None,
            ),
            model_config=NS(processor=None),
            lora_as_adapter=False,
            engine=NS(generate=generate),
            replica_rank=0,
            global_steps=7,
        )
        params = dict(max_tokens=1, temperature=0, top_p=1, top_k=-1, logprobs=False, _latr_top_logprobs=2)
        with patch.dict(sys.modules, {"verl.experimental.latr.vllm_contract": contract}):
            output = await env["generate"](server, [100], params, "probe")
        self.assertEqual(seen[0]["sampling_params"].logprobs, 2)
        self.assertNotIn("_latr_top_logprobs", vars(seen[0]["sampling_params"]))
        self.assertEqual(len(output.extra_fields["latr_candidates"]), 2)
        self.assertEqual(params["_latr_top_logprobs"], 2)  # Caller input stays reusable.
        self.assertEqual(output.extra_fields["global_steps"], 7)
        # Existing ordinary calls must keep their boolean chosen-token contract.
        for want_logprobs, expected in [(False, None), (True, 0)]:
            with patch.dict(sys.modules, {"verl.experimental.latr.vllm_contract": contract}):
                output = await env["generate"](server, [100], {"max_tokens": 1, "logprobs": want_logprobs}, "standard")
            self.assertEqual(seen[-1]["sampling_params"].logprobs, expected)
            self.assertNotIn("latr_candidates", output.extra_fields)


class ExampleTests(unittest.TestCase):
    def launch_args(self, npu=False):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            output = folder / "args.json"
            executable = folder / "python"
            executable.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['LATR_CAPTURE_ARGS'], 'w') as output:\n"
                "    json.dump(sys.argv[1:], output)\n"
            )
            executable.chmod(0o755)
            environment = dict(os.environ, PATH=f"{folder}:{os.environ['PATH']}", LATR_CAPTURE_ARGS=str(output))
            environment.update(NDEVICES_PER_NODE="4", ROLLOUT_TP="2", TRAIN_BATCH_SIZE="8")
            script = "run_grpo_qwen35_9b_npu.sh" if npu else "run_grpo_countdown.sh"
            subprocess.run(["bash", str(ROOT / "examples/latr" / script)], env=environment, check=True)
            return json.loads(output.read_text())[2:]

    def test_shell_arguments_keep_template_and_npu_overrides(self):
        args = self.launch_args()
        template = next(arg for arg in args if arg.startswith("actor_rollout_ref.model.custom_chat_template="))
        self.assertEqual(template.split("=", 1)[1], '"' + data.PASSTHROUGH_CHAT_TEMPLATE + '"')
        args = self.launch_args(npu=True)
        self.assertIn("trainer.device=npu", args)
        self.assertIn("actor_rollout_ref.model.custom_chat_template=null", args)
        self.assertIn("actor_rollout_ref.rollout.enable_prefix_caching=false", args)

    def test_hydra_composes_actual_launch_arguments_and_latr_validation(self):
        try:
            from hydra import compose, initialize_config_dir
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("hydra-core/omegaconf not installed")
        for npu in (False, True):
            with (
                self.subTest(npu=npu),
                initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None),
            ):
                cfg = compose(config_name="ppo_trainer", overrides=self.launch_args(npu))
                config_module.validate_training_config(cfg)
                latr = OmegaConf.to_object(
                    OmegaConf.merge(OmegaConf.structured(LATRConfig), cfg.actor_rollout_ref.rollout.latr)
                )
                self.assertIsInstance(latr, LATRConfig)
                self.assertTrue(latr.enable)
                self.assertEqual(cfg.trainer.device, "npu" if npu else "cuda")
                if npu:
                    self.assertIsNone(cfg.actor_rollout_ref.model.custom_chat_template)
                else:
                    self.assertEqual(cfg.actor_rollout_ref.model.custom_chat_template, data.PASSTHROUGH_CHAT_TEMPLATE)
                for key, value in [
                    ("trainer.use_v1", False),
                    ("trainer.v1.trainer_mode", "async"),
                    ("actor_rollout_ref.rollout.logprobs_mode", "processed_logprobs"),
                    ("actor_rollout_ref.rollout.calculate_log_probs", True),
                    ("algorithm.rollout_correction.bypass_mode", True),
                    ("actor_rollout_ref.rollout.multi_turn.enable", True),
                ]:
                    invalid = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                    OmegaConf.update(invalid, key, value)
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        config_module.validate_training_config(invalid)

    def test_countdown_reward_response_only_and_invalid_expressions(self):
        truth = {"numbers": [1, 2, 3], "target": 9}
        for response, score in [
            ("reasoning\n<answer>(1+2)*3</answer>", 1.0),
            ("<answer>1+2+3</answer>", 0.1),
            ("<answer>9</answer>", 0.1),
            ("no answer", 0.0),
            ("<answer>1**2+3</answer>", 0.1),
        ]:
            with self.subTest(response=response):
                result = reward.compute_score("countdown", response, truth)
                self.assertEqual(result["score"], score)
                self.assertEqual(result["acc"], float(score == 1.0))

    def test_old_prompt_conversion_and_chat_extraction(self):
        raw = "system\nUser: Use 1, 2, 3.\nAssistant: Let me solve this step by step.\n"
        self.assertEqual(data.convert_prompt(raw)[0]["content"], raw)
        self.assertEqual(data.convert_prompt(raw, "chat"), [{"role": "user", "content": "Use 1, 2, 3."}])
        qwen = "system\n<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(data.convert_prompt(qwen, "chat")[0]["content"], "Question")
        with self.assertRaises(ValueError):
            data.convert_prompt("unrecognized template", "chat")


if __name__ == "__main__":
    unittest.main()
