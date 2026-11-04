# Copyright 2026 LATR adaptation contributors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from verl.experimental.latr.cached import generate_trees
from verl.experimental.latr.config import LATRConfig, validate_training_config
from verl.experimental.latr.core import Candidate, TreeState
from verl.experimental.latr.hybrid import finish_group, prepare_batch
from verl.experimental.latr.tree_worker import CachedTreeWorker


def candidates(main=0.5, alternative=0.4):
    return [Candidate(3, math.log(main)), Candidate(4, math.log(alternative))]


def test_pruning_includes_fork_token_and_removes_descendants():
    cfg = LATRConfig(rollout_filter_steps=[2], rollout_filter_edit_dist_thres=0.6)
    state = TreeState(cfg, set())
    state.start(4, 5, 0)
    state.advance({0: candidates()})
    assert len(state.branches) == 2
    state.advance({b.id: candidates() for b in state.active})
    assert state.stats["checks"]["2"] == 1
    # One of two tokens differs => distance .5; child and newborn grandchild die.
    assert {b.id for b in state.branches} == {0, 2}
    assert state.stats["pruned"] == 2


@pytest.mark.parametrize("prob,expected", [(0.25, 1), (0.251, 2)])
def test_strict_raw_probability_threshold(prob, expected):
    state = TreeState(LATRConfig(prob_filter_rel_thres=1, force_return_step=1), set())
    state.start(2, 4, 0)
    state.advance({0: candidates(0.5, prob)})
    assert len(state.branches) == expected
    assert len(state.result().trajectories) == 2


def test_eos_not_forwarded_and_standard_mix():
    state = TreeState(LATRConfig(mix_ratio_initial=0.5), {3})
    state.start(4, 8, 0)
    state.advance({0: [Candidate(3, 0)]})
    assert state.done and not state.active
    result = state.result()
    assert [t.tokens for t in result.trajectories] == [[3], [3], [], []]


def tiny_model(kind):
    torch.manual_seed(7)
    if kind == "llama":
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        cls = LlamaForCausalLM
    else:
        config = Qwen3_5Config(
            text_config=dict(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                linear_conv_kernel_dim=4,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
            ),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=32),
        )
        cls = Qwen3_5ForConditionalGeneration
    config._attn_implementation = "eager"
    return cls(config).eval()


class CheckedForward:
    """Compare every cached row against a real unpadded full-prefix forward.

    Track histories through the actual cache reorder, including duplicated and
    removed rows. This catches KV-only forks on Qwen's recurrent layer.
    """

    def __init__(self, model):
        self.model = model
        self.histories = []
        self.widths = []
        self.reorders = []

    def __call__(self, **kwargs):
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        self.widths.append(ids.shape[1])
        if kwargs.get("past_key_values") is None:
            self.histories = [row[valid.bool()].tolist() for row, valid in zip(ids, mask, strict=True)]
        else:
            self.histories = [h + row.tolist() for h, row in zip(self.histories, ids, strict=True)]
        output = self.model(**kwargs)
        for row, history in enumerate(self.histories):
            full_ids = torch.tensor([history])
            full = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                position_ids=torch.arange(len(history)).unsqueeze(0),
                use_cache=False,
                logits_to_keep=1,
            )
            torch.testing.assert_close(output.logits[row, -1], full.logits[0, -1], atol=2e-5, rtol=2e-4)
        cache = output.past_key_values
        if not getattr(cache, "_latr_checked", False):
            original = cache.reorder_cache

            def reorder(indices):
                self.reorders.append(indices.tolist())
                self.histories = [self.histories[i].copy() for i in indices.tolist()]
                original(indices)

            cache.reorder_cache = reorder
            cache._latr_checked = True
        return output


@pytest.mark.parametrize("kind", ["llama", "qwen35"])
@pytest.mark.parametrize("prune", [False, True])
def test_real_cached_logits_match_full_prefix_after_fork_and_prune(kind, prune):
    model = CheckedForward(tiny_model(kind))
    config = LATRConfig(
        prob_filter_abs_thres=0,
        prob_filter_rel_thres=1,
        rollout_filter_steps=[2, 3],
        rollout_filter_edit_dist_thres=0.8 if prune else 0,
        force_return_step=5,
    )
    requests = [
        dict(n=4, response_length=8, global_step=0, prompt_ids=[1, 2, 3]),
        dict(n=3, response_length=8, global_step=0, prompt_ids=[2, 4]),
    ]
    results = generate_trees(model, SimpleNamespace(pad_token_id=0, decode=str), config, requests, set(), "cpu")
    assert [len(r.trajectories) for r in results] == [4, 3]
    assert model.widths[0] == 3 and all(w == 1 for w in model.widths[1:])
    assert any(len(set(rows)) < len(rows) for rows in model.reorders)
    if prune:
        assert sum(r.stats["pruned"] for r in results) > 0


def test_phase_order_and_group_coverage(monkeypatch):
    from tensordict import TensorDict

    calls = []
    payload = dict(index=0)
    request = dict(index=0, n=2, response_length=5, global_step=0, prompt_ids=[1])
    trainer = SimpleNamespace(
        agent_loop_manager=SimpleNamespace(prepare_latr_prompts=lambda batch: [request]),
        checkpoint_manager=SimpleNamespace(
            sleep_replicas=lambda: calls.append("sleep"),
            update_weights=lambda version: calls.append(("vllm", version)),
        ),
        actor_rollout_wg=SimpleNamespace(
            generate_latr_trees=lambda req, version: calls.append(("tree", version)) or [[payload], []]
        ),
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(latr=asdict(LATRConfig())))),
        _latr_policy_version=6,
        timing_raw={},
    )
    batch = TensorDict({}, batch_size=[2])
    prepare_batch(trainer, batch)
    assert calls == ["sleep", ("tree", 6), ("vllm", 6)]
    from verl.utils import tensordict_utils as tu

    assert tu.get(batch, "latr_payload") == [payload, None]
    trainer.actor_rollout_wg.generate_latr_trees = lambda *args: [[]]
    calls.clear()
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        prepare_batch(trainer, TensorDict({}, batch_size=[2]))
    assert calls == ["sleep"]  # failed tree phase must never wake vLLM


def test_continuations_keep_prefix_masks_and_skip_eos():
    calls, stored = [], []

    async def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            token_ids=[8, 9], stop_reason="completed", extra_fields=dict(min_global_steps=2, max_global_steps=2)
        )

    async def postprocess(output, validate, **kwargs):
        stored.append(output)

    worker = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(response_length=5))),
        llm_client=SimpleNamespace(generate=generate),
        _agent_loop_postprocess=postprocess,
    )
    payload = dict(
        prompt_ids=[1, 2],
        version=2,
        eos_ids=[9],
        stats={},
        trajectories=[dict(tokens=[3, 4]), dict(tokens=[9]), dict(tokens=[])],
    )
    asyncio.run(finish_group(worker, payload, dict(temperature=1), dict(uid="x"), False))
    assert sorted(o.response_ids for o in stored) == [[3, 4, 8, 9], [8, 9], [9]]
    assert all(o.prompt_ids == [1, 2] and o.response_mask == [1] * len(o.response_ids) for o in stored)
    assert sorted(c["sampling_params"]["max_tokens"] for c in calls) == [3, 5]
    assert all(o.response_logprobs is None for o in stored)
    payload["version"] = 3
    with pytest.raises(RuntimeError, match="policy version mismatch"):
        asyncio.run(finish_group(worker, payload, {}, dict(uid="x"), False))


def test_weight_coverage_and_failure_offload(monkeypatch):
    worker = CachedTreeWorker.__new__(CachedTreeWorker)
    worker.parameters = dict(a=torch.nn.Parameter(torch.zeros(2)))
    worker.buffers, worker.received = {}, set()
    worker.model = torch.nn.Module()
    worker.model.register_parameter("a", worker.parameters["a"])
    worker.begin_weights()
    with pytest.raises(ValueError, match="did not receive"):
        worker.finish_weights(1)
    worker.load_bucket([("a", torch.ones(2))])
    worker.finish_weights(1)
    assert torch.equal(worker.parameters["a"], torch.ones(2))
    worker.model = SimpleNamespace(to=lambda device: None, cpu=lambda: calls.append("cpu"))
    calls = []
    worker.device = "cpu"
    worker.device_api = SimpleNamespace(
        synchronize=lambda: calls.append("sync"), empty_cache=lambda: calls.append("empty")
    )
    worker.config = LATRConfig()
    worker.tokenizer, worker.eos_ids = None, set()

    def fail(*args):
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr("verl.experimental.latr.cached.generate_trees", fail)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        worker.generate([{}], 1)
    assert calls == ["cpu", "sync", "empty"]


def test_composed_configuration_and_guards():
    from hydra import compose, initialize_config_module

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.n=4",
                "actor_rollout_ref.rollout.calculate_log_probs=false",
                "actor_rollout_ref.rollout.latr.enable=true",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.fsdp_config.offload_policy=true",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
                "algorithm.adv_estimator=grpo",
                "trainer.use_v1=true",
                "trainer.v1.trainer_mode=sync",
            ],
        )
    validate_training_config(cfg)
    cfg.actor_rollout_ref.rollout.calculate_log_probs = True
    with pytest.raises(ValueError, match="actor recomputes"):
        validate_training_config(cfg)
    assert OmegaConf.to_container(cfg.actor_rollout_ref.rollout.latr) == dict(asdict(LATRConfig(enable=True)))


def test_zero_tree_ratio_skips_hf_weight_export():
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    config = LATRConfig(mix_ratio_initial=0)
    calls = []
    trainer = SimpleNamespace(
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(latr=asdict(config)))),
        agent_loop_manager=SimpleNamespace(
            prepare_latr_prompts=lambda batch: [dict(index=0, n=2, response_length=5, global_step=9, prompt_ids=[1])]
        ),
        checkpoint_manager=SimpleNamespace(
            sleep_replicas=lambda: calls.append("sleep"), update_weights=lambda v: calls.append("vllm")
        ),
        actor_rollout_wg=None,
        _latr_policy_version=8,
        timing_raw={},
    )
    batch = TensorDict({}, batch_size=[1])
    prepare_batch(trainer, batch)
    assert calls == ["sleep", "vllm"]
    assert len(tu.get(batch, "latr_payload")[0]["trajectories"]) == 2


def test_export_error_drains_all_collectives_and_keeps_vllm_asleep(monkeypatch):
    import ray

    from verl.workers.engine_workers import ActorRolloutRefWorker

    calls = []

    def params(**kwargs):
        assert kwargs["raw_hf_keys"]

        def stream():
            for i in range(3):
                calls.append(("gather", i))
                yield f"p{i}", torch.ones(600_000)

        return stream(), None

    def fail_bucket(bucket):
        calls.append("bucket")
        raise RuntimeError("bad weight")

    remote = lambda fn: SimpleNamespace(remote=fn)
    fake = SimpleNamespace(
        rank=0,
        world_size=1,
        config=SimpleNamespace(rollout=SimpleNamespace(latr=OmegaConf.create(asdict(LATRConfig(weight_bucket_mb=1))))),
        _latr_tree_worker=SimpleNamespace(
            begin_weights=remote(lambda: {f"p{i}": torch.bfloat16 for i in range(3)}),
            load_bucket=remote(fail_bucket),
        ),
        actor=SimpleNamespace(
            engine=SimpleNamespace(get_per_tensor_param=params, to=lambda *a, **kw: calls.append("offload"))
        ),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(torch.distributed, "all_gather_object", lambda output, error: output.__setitem__(0, error))
    monkeypatch.setattr("verl.workers.engine_workers.aggressive_empty_cache", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="bad weight"):
        ActorRolloutRefWorker.generate_latr_trees(fake, [{}], 0)
    assert calls == [("gather", 0), "bucket", ("gather", 1), ("gather", 2), "offload"]


@pytest.mark.parametrize("script", ["run_grpo_countdown.sh", "run_grpo_qwen35_9b_npu.sh"])
def test_launch_scripts_compose_valid_hybrid_config(tmp_path, script):
    import json

    from hydra import compose, initialize_config_module

    # Capture the exact shell-expanded argv without launching Ray or loading a
    # model; Hydra still validates all overrides, including the NPU wrapper.
    fake_python = tmp_path / "python"
    fake_python.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "open(os.environ['LATR_CAPTURE_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    fake_python.chmod(0o755)
    capture = tmp_path / "args.json"
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", LATR_CAPTURE_ARGS=str(capture))
    env.update(NDEVICES_PER_NODE="2", ROLLOUT_TP="1", MODEL_PATH="/synthetic/model", TRAIN_BATCH_SIZE="8")
    repo = Path(__file__).resolve().parents[3]
    subprocess.run(["bash", str(repo / "examples/latr" / script)], env=env, check=True)
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", "verl.trainer.main_ppo"]
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=argv[2:])
    validate_training_config(cfg)

    from verl.utils.config import omega_conf_to_dataclass

    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    assert isinstance(rollout.latr, LATRConfig) and rollout.latr.enable


def test_tree_worker_loads_checkpoint_and_refreshes_after_rollout(tmp_path, monkeypatch):
    from verl.utils import device as device_utils

    model = tiny_model("qwen35")
    model.save_pretrained(tmp_path)
    monkeypatch.setattr(
        device_utils,
        "get_torch_device",
        lambda: SimpleNamespace(set_device=lambda i: None, synchronize=lambda: None, empty_cache=lambda: None),
    )
    config = LATRConfig(force_return_step=2, attn_implementation="eager")
    worker = CachedTreeWorker(
        SimpleNamespace(
            tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=31, decode=str),
            hf_config=model.config,
            local_path=str(tmp_path),
            trust_remote_code=False,
            generation_config=model.generation_config,
        ),
        asdict(config),
        "float32",
        "cpu",
    )
    for version in [0, 1]:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.01)
        worker.begin_weights()
        worker.load_bucket(list(model.state_dict().items()))
        worker.finish_weights(version)
        results = worker.generate([dict(index=0, n=2, response_length=6, global_step=0, prompt_ids=[1, 2])], version)
        assert results[0]["version"] == version and len(results[0]["trajectories"]) == 2
        for name, parameter in worker.model.named_parameters():
            torch.testing.assert_close(parameter, dict(model.named_parameters())[name])


def test_sync_trainer_defers_wake_and_preserves_policy_version():
    from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync

    calls = []
    trainer = SimpleNamespace(
        global_steps=7,
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(latr=LATRConfig(enable=True)))
        ),
        checkpoint_manager=SimpleNamespace(
            sleep_replicas=lambda: calls.append("sleep"),
            update_weights=lambda v: calls.append(("update", v)),
        ),
    )
    PPOTrainerSync.on_init_end(trainer)
    assert trainer._latr_policy_version == 7 and not calls
    trainer.global_steps = 8
    assert trainer._latr_policy_version == 7  # sampling sees the preceding completed update
    PPOTrainerSync.on_step_end(trainer)
    assert trainer._latr_policy_version == 8 and not calls
    PPOTrainerSync.on_validate_begin(trainer)
    assert calls == ["sleep", ("update", 8)]
