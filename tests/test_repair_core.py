from __future__ import annotations

import json

import numpy as np
import pytest

from drone6dof.connectome import RateCodedConnectomeSNN, SparseRecurrence
from drone6dof.repair.activity import audit_bundle
from drone6dof.repair.artifacts import (
    create_run_dir,
    load_cycle,
    load_run,
    snapshot_resume_round,
    write_baseline,
    write_cycle,
    write_final,
    write_json,
)
from drone6dof.repair.edits import (
    apply_program,
    forward_actions,
    prune_program_units,
    save_repaired_bundle,
    validate_candidate,
)
from drone6dof.repair import scenarios as scenarios_mod
from drone6dof.repair.loop import (
    NoShiftError,
    RepairLoop,
    RepairRunConfig,
    ResumeError,
    _accept_candidate,
)
from drone6dof.repair.proposers.base import ProposalContext
from drone6dof.repair.proposers.llm import (
    LLMConfig,
    LLMProposer,
    OpenAIClient,
    extract_json,
)
from drone6dof.repair.proposers.llm_tools import TOOL_SCHEMA, RepairToolbox
from drone6dof.repair.scenarios import REGISTRY, evaluate_bundle
from drone6dof.repair.types import (
    Budget,
    CycleRecord,
    Diagnosis,
    EditOp,
    EditProgram,
    EpisodeResult,
    SplitResult,
    ValidationReport,
)
from drone6dof.weights import (
    default_fields,
    fingerprint,
    load_weights,
    save_weights,
)


def _bundle():
    return load_weights("weights/quad6dof_connectome.npz", expect_fields=default_fields())


def test_bias_zero_is_backward_compatible():
    b = _bundle()
    edges = np.asarray(b["edges"], dtype=np.int64)
    signed = np.abs(b["w_mag"]) * np.asarray(b["polarity"])[edges[1]]
    rec = SparseRecurrence(edges, signed, int(b["n_neurons"]))
    a = RateCodedConnectomeSNN(b["w_in"], b["w_out"], rec, micro_steps=10, v_th=1.0)
    z = RateCodedConnectomeSNN(
        b["w_in"], b["w_out"], rec, micro_steps=10, v_th=1.0,
        bias=np.zeros(int(b["n_neurons"])),
    )
    x = np.zeros(int(b["w_in"].shape[1]), dtype=np.float64)
    x[:3] = [0.3, -0.2, 0.1]
    np.testing.assert_allclose(a.forward_input(x), z.forward_input(x), rtol=0, atol=0)


def test_spike_counts_accumulate():
    b = _bundle()
    edges = np.asarray(b["edges"], dtype=np.int64)
    signed = np.abs(b["w_mag"]) * np.asarray(b["polarity"])[edges[1]]
    net = RateCodedConnectomeSNN(b["w_in"], b["w_out"], SparseRecurrence(edges, signed, int(b["n_neurons"])))
    x = np.zeros(int(b["w_in"].shape[1]), dtype=np.float64)
    for _ in range(5):
        net.forward_input(x)
    assert net.frames == 5
    assert net.spike_counts.shape[0] == int(b["n_neurons"])
    assert net.spike_frames.sum() <= net.spike_counts.sum()


def test_episode_is_deterministic():
    b = _bundle()
    scenario = REGISTRY["nominal"]
    a = evaluate_bundle(scenario, b, [7], steps=20)
    c = evaluate_bundle(scenario, b, [7], steps=20)
    assert a.metrics() == c.metrics()
    assert a.episodes[0].seed == c.episodes[0].seed == 7


def _activity_and_features():
    b = _bundle()
    nominal = REGISTRY["nominal"]
    split = evaluate_bundle(nominal, b, [0], steps=30, capture=True)
    activity = audit_bundle(
        b,
        nominal,
        steps=30,
        traces=[e.trace for e in split.episodes if e.trace],
    )
    return b, activity


def _silent_on_demo(b, demo):
    edges = np.asarray(b["edges"], dtype=np.int64)
    signed = np.abs(b["w_mag"]) * np.asarray(b["polarity"])[edges[1]]
    net = RateCodedConnectomeSNN(
        b["w_in"], b["w_out"], SparseRecurrence(edges, signed, int(b["n_neurons"]))
    )
    net.reset()
    for row in demo:
        net.forward_input(row)
    return set(int(u) for u in np.flatnonzero(net.spike_counts == 0))


def test_quiet_gated_program_and_bundle_roundtrip(tmp_path):
    b, activity = _activity_and_features()
    n_in = int(b["w_in"].shape[1])
    demo = np.zeros((5, n_in), dtype=np.float64)
    demo[:, 41] = 1.0
    spare = [
        int(u)
        for u in np.flatnonzero(activity.spare).astype(int).tolist()
        if int(u) in _silent_on_demo(b, demo)
    ]
    assert len(spare) >= 2
    gate, action = spare[0], spare[1]
    weights = np.zeros(n_in, dtype=np.float64)
    weights[41] = -1.0  # min_range, low direction
    program = EditProgram(
        program_id="test-quiet",
        ops=[
            EditOp(kind="silence_output", unit=gate),
            EditOp(kind="isolate_unit", unit=gate),
            EditOp(kind="isolate_unit", unit=action),
            EditOp(kind="set_input_row", unit=gate, values=weights.tolist()),
            EditOp(kind="set_bias", unit=gate, value=0.5),
            EditOp(kind="set_input_row", unit=action, values=[0.0] * n_in),
            EditOp(kind="set_bias", unit=action, value=-0.2),
            EditOp(kind="add_edge", src=gate, dst=action, magnitude=1.0),
            EditOp(kind="set_output_col", unit=action, values=[0.0, 0.0, 0.1]),
        ],
        rationale="synthetic quiet gate",
    )
    target = np.zeros((5, n_in), dtype=np.float64)
    target[:, 41] = 0.0
    candidate, report = apply_program(
        b,
        program,
        budget=Budget(max_param_change_per_cycle=100000, max_units_per_cycle=8),
        spare_units=spare,
    )
    assert report.ok, report.issues
    report = validate_candidate(
        b,
        candidate,
        program,
        report,
        demo_features=demo,
        target_features=target,
        budget=Budget(quiet_eps=1e-9),
    )
    assert report.ok, report.issues
    assert report.quietness["max_abs_delta"] == 0.0
    assert report.liveness["max_rate_hz"] > 0.0
    assert report.changed_params < 5000

    path = tmp_path / "repaired.npz"
    save_repaired_bundle(path, candidate, repair={"run_id": "test"})
    reloaded = load_weights(path, expect_fields=default_fields())
    assert reloaded["fingerprint"] == fingerprint(default_fields())
    assert reloaded.get("repair", {}).get("run_id") == "test"
    np.testing.assert_allclose(reloaded["w_in"], candidate["w_in"])
    np.testing.assert_allclose(reloaded["bias"], candidate["bias"])


def test_json_program_parsing():
    data = {
        "ops": [
            {"kind": "set_bias", "unit": 3, "value": 0.2, "rationale": "x"},
            {"type": "set_output_col", "unit": 3, "values": [0, 0, 0.1]},
        ],
        "rationale": "hello",
    }
    parsed = extract_json("```json\n" + json.dumps(data) + "\n```")
    program = EditProgram.from_dict(
        {
            "program_id": "p",
            "ops": parsed["ops"],
            "rationale": parsed["rationale"],
        }
    )
    assert len(program.ops) == 2
    assert program.ops[1].kind == "set_output_col"

    aliased = EditProgram.from_dict(
        {
            "program_id": "p2",
            "ops": [{"op": "set_bias", "unit": 3, "value": 0.1}],
        }
    )
    assert aliased.ops[0].kind == "set_bias"


def test_llm_proposer_with_fake_client():
    b, activity = _activity_and_features()
    spare = np.flatnonzero(activity.spare).astype(int).tolist()
    n_in = int(b["w_in"].shape[1])
    program = {
        "ops": [
            {"kind": "set_input_row", "unit": spare[0], "values": [0.0] * n_in},
            {"kind": "set_bias", "unit": spare[0], "value": 0.0},
        ],
        "rationale": "no-op from fake model",
    }

    class Fake:
        def complete(self, messages, tools=None):
            return {"role": "assistant", "content": json.dumps(program)}

    scenario = REGISTRY["nominal"]
    diagnosis = Diagnosis(
        scenario=scenario.name,
        split_metrics={},
        failure_counts={},
        activity=activity.summary,
        examples=[],
    )
    context = ProposalContext(
        cycle=1,
        scenario=scenario,
        budget=Budget(max_turns=1),
        activity=activity,
        diagnosis=diagnosis,
        demo_features=np.zeros((2, n_in)),
        target_features=np.zeros((2, n_in)),
        base_bundle=b,
    )
    out = LLMProposer(client=Fake()).propose(context)
    assert out.ops and out.proposer == "llm"


def _clear_llm_env(monkeypatch):
    for name in (
        "REPAIR_LLM_PROVIDER",
        "REPAIR_LLM_API_KEY",
        "REPAIR_LLM_BASE_URL",
        "REPAIR_LLM_MODEL",
        "REPAIR_LLM_TIMEOUT",
        "REPAIR_LLM_TEMPERATURE",
        "REPAIR_LLM_MAX_TOKENS",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_llm_config_deepseek_defaults(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    cfg = LLMConfig.from_env(provider="deepseek")
    assert cfg.provider == "deepseek"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.api_key == "deepseek-key"
    assert cfg.model == "deepseek-chat"
    assert cfg.supports_tools is True
    assert cfg.supports_sampling is True


def test_llm_config_deepseek_reasoner_flags(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    cfg = LLMConfig.from_env(provider="deepseek")
    assert cfg.model == "deepseek-reasoner"
    assert cfg.supports_tools is False
    assert cfg.supports_sampling is False


def test_llm_config_deepseek_generic_fallbacks(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("REPAIR_LLM_API_KEY", "generic-key")
    monkeypatch.setenv("REPAIR_LLM_BASE_URL", "https://proxy.example/v1")
    cfg = LLMConfig.from_env(provider="deepseek")
    assert cfg.api_key == "generic-key"
    assert cfg.base_url == "https://proxy.example/v1"
    assert cfg.model == "deepseek-chat"


def test_llm_config_provider_and_model_precedence(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("REPAIR_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    cfg = LLMConfig.from_env(provider="deepseek")
    assert cfg.provider == "deepseek"
    assert cfg.model == "deepseek-chat"
    explicit = LLMConfig.from_env(provider="deepseek", model="custom-deepseek")
    assert explicit.model == "custom-deepseek"


def test_llm_payload_reasoner_omits_tools_and_temperature():
    cfg = LLMConfig(
        base_url="https://api.deepseek.com",
        api_key="x",
        model="deepseek-reasoner",
        provider="deepseek",
        supports_tools=False,
        supports_sampling=False,
    )
    client = OpenAIClient(cfg)
    payload = client.build_payload(
        [{"role": "user", "content": "hello"}], tools=TOOL_SCHEMA
    )
    assert "tools" not in payload
    assert "tool_choice" not in payload
    assert "temperature" not in payload
    assert payload["model"] == "deepseek-reasoner"


def test_llm_payload_chat_keeps_tools_and_temperature():
    cfg = LLMConfig(
        base_url="https://api.deepseek.com",
        api_key="x",
        model="deepseek-chat",
        provider="deepseek",
    )
    payload = OpenAIClient(cfg).build_payload(
        [{"role": "user", "content": "hello"}], tools=TOOL_SCHEMA
    )
    assert payload["tools"] == TOOL_SCHEMA
    assert payload["tool_choice"] == "auto"
    assert payload["temperature"] == cfg.temperature


def test_llm_payload_strips_reasoning_content():
    cfg = LLMConfig(api_key="x", provider="deepseek", supports_tools=True)
    payload = OpenAIClient(cfg).build_payload(
        [
            {
                "role": "assistant",
                "content": "final",
                "reasoning_content": "private chain",
                "tool_calls": [],
            }
        ]
    )
    assert "reasoning_content" not in payload["messages"][0]
    assert payload["messages"][0]["content"] == "final"


def test_llm_missing_deepseek_key_is_actionable():
    cfg = LLMConfig(
        base_url="https://api.deepseek.com",
        api_key="",
        model="deepseek-chat",
        provider="deepseek",
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        OpenAIClient(cfg).complete([{"role": "user", "content": "x"}])


def test_llm_proposer_reasoner_uses_json_without_tools():
    b, activity = _activity_and_features()
    spare = np.flatnonzero(activity.spare).astype(int).tolist()
    n_in = int(b["w_in"].shape[1])
    program = {
        "ops": [
            {"kind": "set_input_row", "unit": spare[0], "values": [0.0] * n_in},
            {"kind": "set_bias", "unit": spare[1], "value": 0.0},
        ],
        "rationale": "reasoner JSON fallback",
    }

    class ReasonerFake:
        def __init__(self):
            self.tools = "unset"

        def complete(self, messages, tools=None):
            self.tools = tools
            return {"role": "assistant", "content": json.dumps(program)}

    fake = ReasonerFake()
    cfg = LLMConfig(
        base_url="https://api.deepseek.com",
        api_key="x",
        model="deepseek-reasoner",
        provider="deepseek",
        supports_tools=False,
        supports_sampling=False,
    )
    scenario = REGISTRY["nominal"]
    diagnosis = Diagnosis(
        scenario=scenario.name,
        split_metrics={},
        failure_counts={},
        activity=activity.summary,
        examples=[],
    )
    context = ProposalContext(
        cycle=1,
        scenario=scenario,
        budget=Budget(max_turns=1),
        activity=activity,
        diagnosis=diagnosis,
        demo_features=np.zeros((2, n_in)),
        target_features=np.zeros((2, n_in)),
        base_bundle=b,
    )
    out = LLMProposer(config=cfg, client=fake).propose(context)
    assert fake.tools is None
    assert out.proposer == "llm"
    assert out.metadata.get("provider") == "deepseek"


def test_llm_proposer_tool_call_submission():
    b, activity = _activity_and_features()
    spare = np.flatnonzero(activity.spare).astype(int).tolist()
    n_in = int(b["w_in"].shape[1])
    arguments = json.dumps({
        "ops": [
            {"kind": "set_input_row", "unit": spare[0], "values": [0.0] * n_in},
            {"kind": "set_bias", "unit": spare[0], "value": 0.0},
        ],
        "rationale": "submitted through the tool",
    })

    class ToolFake:
        def complete(self, messages, tools=None):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "submit_program", "arguments": arguments},
                    }
                ],
            }

    scenario = REGISTRY["nominal"]
    diagnosis = Diagnosis(
        scenario=scenario.name,
        split_metrics={},
        failure_counts={},
        activity=activity.summary,
        examples=[],
    )
    context = ProposalContext(
        cycle=1,
        scenario=scenario,
        budget=Budget(max_turns=1),
        activity=activity,
        diagnosis=diagnosis,
        demo_features=np.zeros((2, n_in)),
        target_features=np.zeros((2, n_in)),
        base_bundle=b,
    )
    out = LLMProposer(client=ToolFake()).propose(context)
    assert out.proposer == "llm"
    assert out.ops[1].kind == "set_bias"
    assert "submitted through" in out.rationale


def _episode(name: str) -> EpisodeResult:
    return EpisodeResult(
        episode_id=0,
        seed=1,
        scenario=name,
        success=True,
        reached_goal=True,
        collision_free=True,
        final_goal_dist_m=0.1,
        closest_goal_dist_m=0.1,
        clearance_min_m=0.2,
        collisions=0,
        steps=10,
        time_to_goal_s=1.0,
        progress=0.9,
        command_saturation_frac=0.0,
        mean_power_w=100.0,
        failure="none",
    )


def _probe_episode(seed: int, near: bool, progress: float, collision_free: bool = True):
    return EpisodeResult(
        episode_id=seed,
        seed=seed,
        scenario="probe",
        success=near,
        reached_goal=near,
        collision_free=collision_free,
        final_goal_dist_m=0.4 if near else 2.0,
        closest_goal_dist_m=0.4 if near else 2.0,
        clearance_min_m=0.1 if collision_free else -0.1,
        collisions=0 if collision_free else 1,
        steps=10,
        time_to_goal_s=1.0 if near else None,
        progress=progress,
        final_progress=progress,
        near_success=near,
        failure="none" if near else "stall_short",
    )


def _probe_split(name, near_flags, progress, collision_free=True):
    return SplitResult(
        scenario=name,
        split="probe",
        steps=10,
        episodes=[
            _probe_episode(i, near, progress, collision_free)
            for i, near in enumerate(near_flags)
        ],
    )


def _probe_summary(name="goal_behind_wall", gap=1.0, eligible=False):
    return {
        "scenario": name,
        "priority": 10,
        "snn": {
            "near_success_rate": 0.0,
            "mean_final_progress": -0.1,
            "collision_free_rate": 1.0,
        },
        "teacher": {
            "near_success_rate": 0.667,
            "mean_final_progress": 0.9,
            "collision_free_rate": 1.0,
        },
        "gap": gap,
        "eligible": eligible,
        "reasons": [] if eligible else ["teacher floors"],
    }


def test_auto_select_tolerates_one_noisy_seed(monkeypatch):
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_bundle",
        lambda scenario, bundle, seeds, **kwargs: _probe_split(
            scenario.name, [False, False, False], -0.1
        ),
    )
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_teacher",
        lambda scenario, seeds, **kwargs: _probe_split(
            scenario.name, [True, True, False], 0.9
        ),
    )
    scenario, summaries = scenarios_mod.auto_select_shift(
        {},
        scenarios=[REGISTRY["goal_behind_wall"]],
        probe_seeds=[1, 2, 3],
        probe_steps=10,
    )
    assert scenario is not None
    assert scenario.name == "goal_behind_wall"
    assert summaries[0]["eligible"] is True
    assert summaries[0]["selected"] is True
    assert len(summaries[0]["teacher_episodes"]) == 3


def test_auto_select_accepts_one_near_seed_with_high_progress(monkeypatch):
    """The real 51000..51002 probe: teacher 1/3 near, ~0.91 progress."""
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_bundle",
        lambda scenario, bundle, seeds, **kwargs: _probe_split(
            scenario.name, [False, False, False], -0.06
        ),
    )
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_teacher",
        lambda scenario, seeds, **kwargs: _probe_split(
            scenario.name, [True, False, False], 0.915
        ),
    )
    scenario, summaries = scenarios_mod.auto_select_shift(
        {},
        scenarios=[REGISTRY["goal_behind_wall"]],
        probe_seeds=[51000, 51001, 51002],
        probe_steps=500,
    )
    assert scenario is not None and scenario.name == "goal_behind_wall"
    assert summaries[0]["eligible"] is True
    assert summaries[0]["gap"] > 0.5


def test_auto_select_picks_largest_gap(monkeypatch):
    def fake_snn(scenario, bundle, seeds, **kwargs):
        return _probe_split(scenario.name, [False, False, False], -0.1)

    def fake_teacher(scenario, seeds, **kwargs):
        if scenario.name == "goal_behind_wall":
            return _probe_split(scenario.name, [True, True, True], 0.9)
        return _probe_split(scenario.name, [True, True, True], 0.6)

    monkeypatch.setattr(scenarios_mod, "evaluate_bundle", fake_snn)
    monkeypatch.setattr(scenarios_mod, "evaluate_teacher", fake_teacher)
    scenario, summaries = scenarios_mod.auto_select_shift(
        {},
        scenarios=[REGISTRY["wall"], REGISTRY["goal_behind_wall"]],
        probe_seeds=[1, 2, 3],
        probe_steps=10,
        strong_gap=99.0,
    )
    assert scenario is not None and scenario.name == "goal_behind_wall"
    selected = [s for s in summaries if s["selected"]]
    assert len(selected) == 1 and selected[0]["scenario"] == "goal_behind_wall"


def test_auto_select_reports_no_eligible_scenario(monkeypatch):
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_bundle",
        lambda scenario, bundle, seeds, **kwargs: _probe_split(
            scenario.name, [False, False, False], -0.1
        ),
    )
    monkeypatch.setattr(
        scenarios_mod,
        "evaluate_teacher",
        lambda scenario, seeds, **kwargs: _probe_split(
            scenario.name, [False, False, False], 0.9
        ),
    )
    scenario, summaries = scenarios_mod.auto_select_shift(
        {},
        scenarios=[REGISTRY["goal_behind_wall"]],
        probe_seeds=[1, 2, 3],
        probe_steps=10,
    )
    assert scenario is None
    assert summaries[0]["eligible"] is False
    assert any("teacher floors" in reason for reason in summaries[0]["reasons"])


def test_no_shift_writes_probe_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "drone6dof.repair.loop.auto_select_shift",
        lambda *args, **kwargs: (None, [_probe_summary(eligible=False)]),
    )
    config = RepairRunConfig(
        bundle_path="weights/quad6dof_connectome.npz",
        out_root=str(tmp_path),
        run_id="no-shift",
        scenario="auto",
        proposer="template",
        probe_episodes=1,
        probe_steps=10,
    )
    with pytest.raises(NoShiftError) as exc:
        RepairLoop(config).run()
    run_dir = tmp_path / "no-shift"
    assert (run_dir / "probe.json").exists()
    assert (run_dir / "probe.md").exists()
    assert (run_dir / "report.md").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "no_shift"
    assert manifest["suggested_scenario"] == "goal_behind_wall"
    assert "probe" in str(exc.value).lower()


def test_auto_isolation_without_explicit_op():
    b, activity = _activity_and_features()
    degree = activity.in_degree + activity.out_degree
    candidates = [
        int(u)
        for u in np.flatnonzero(activity.spare & (degree > 0)).tolist()
    ]
    assert candidates
    unit = candidates[0]
    n_in = int(b["w_in"].shape[1])
    program = EditProgram(
        "auto-isolate",
        [
            EditOp(kind="set_input_row", unit=unit, values=[0.0] * n_in),
            EditOp(kind="set_bias", unit=unit, value=0.0),
            EditOp(kind="set_output_col", unit=unit, values=[0.0, 0.0, 0.0]),
        ],
    )
    candidate, report = apply_program(
        b,
        program,
        budget=Budget(max_param_change_per_cycle=100000),
        spare_units=candidates,
    )
    assert unit in report.auto_isolated_units
    edges = np.asarray(candidate["edges"], dtype=np.int64)
    mask = (edges[0] == unit) | (edges[1] == unit)
    assert np.all(np.asarray(candidate["w_mag"])[mask] == 0.0)


def test_quietness_is_a_warning_by_default():
    b, activity = _activity_and_features()
    spare = np.flatnonzero(activity.spare).astype(int).tolist()
    unit = spare[0]
    n_in = int(b["w_in"].shape[1])
    program = EditProgram(
        "noisy",
        [
            EditOp(kind="set_input_row", unit=unit, values=[1.0] * n_in),
            EditOp(kind="set_bias", unit=unit, value=1.5),
            EditOp(kind="set_output_col", unit=unit, values=[1.0, 1.0, 1.0]),
        ],
    )
    candidate, report = apply_program(
        b,
        program,
        budget=Budget(max_param_change_per_cycle=100000),
        spare_units=spare,
    )
    report = validate_candidate(
        b,
        candidate,
        program,
        report,
        demo_features=np.zeros((3, n_in)),
        budget=Budget(),
    )
    assert report.ok is True
    assert report.quietness["max_abs_delta"] > 0.0
    assert any("quietness warning" in warning for warning in report.warnings)

    strict = validate_candidate(
        b,
        candidate,
        program,
        report,
        demo_features=np.zeros((3, n_in)),
        budget=Budget(),
        require_quiet=True,
    )
    assert strict.ok is False
    assert any("quietness failed" in issue for issue in strict.issues)


def test_accept_candidate_is_monotonic_with_guards():
    baseline = _probe_split("goal", [False, False, False], 0.10)
    better = _probe_split("goal", [False, False, False], 0.11)
    equal = _probe_split("goal", [False, False, False], 0.10)
    base_reg = _probe_split("nominal", [False, False, False], 0.10)
    same_reg = _probe_split("nominal", [False, False, False], 0.10)
    bad_reg = _probe_split("nominal", [False, False, False], 0.01)
    collided = _probe_split("goal", [False, False, False], 0.11, collision_free=False)

    assert _accept_candidate(better, same_reg, baseline, base_reg, 0.10, 1e-6)
    assert not _accept_candidate(equal, same_reg, baseline, base_reg, 0.10, 1e-6)
    assert not _accept_candidate(better, bad_reg, baseline, base_reg, 0.10, 1e-6)
    assert not _accept_candidate(collided, same_reg, baseline, base_reg, 0.10, 1e-6)


def test_validate_program_tool_reports_warnings_without_rollouts():
    b, activity = _activity_and_features()
    spare = np.flatnonzero(activity.spare).astype(int).tolist()
    n_in = int(b["w_in"].shape[1])
    toolbox = RepairToolbox(
        b,
        activity,
        budget=Budget(max_param_change_per_cycle=100000),
        demo_features=np.zeros((3, n_in)),
        target_features=np.zeros((3, n_in)),
        spare_units=spare,
    )
    result = toolbox.call(
        "validate_program",
        {
            "ops": [
                {"kind": "set_input_row", "unit": spare[0], "values": [0.0] * n_in},
                {"kind": "set_bias", "unit": spare[0], "value": 0.0},
                {"kind": "set_output_col", "unit": spare[0], "values": [0.0, 0.0, 0.0]},
            ],
            "rationale": "quiet no-op",
        },
    )
    assert result["ok"] is True
    assert spare[0] in result["auto_isolated_units"]
    assert "quietness" in result
    assert toolbox.practice_runs == 0
    assert toolbox.last_validated is not None


def test_prune_program_units_removes_all_ops_for_unit():
    program = EditProgram(
        "p",
        [
            EditOp(kind="set_bias", unit=1, value=0.1),
            EditOp(kind="set_bias", unit=2, value=0.2),
            EditOp(kind="add_edge", src=1, dst=2, magnitude=1.0),
        ],
    )
    pruned = prune_program_units(program, [2])
    assert all(op.unit != 2 and op.src != 2 and op.dst != 2 for op in pruned.ops)
    assert pruned.metadata["pruned_units"] == [2]


def test_program_subsets_prefers_single_units_first():
    program = EditProgram(
        "p",
        [
            EditOp(kind="set_bias", unit=1, value=0.1),
            EditOp(kind="set_bias", unit=2, value=0.2),
            EditOp(kind="set_bias", unit=3, value=0.3),
            EditOp(kind="add_edge", src=1, dst=2, magnitude=1.0),
        ],
    )
    from drone6dof.repair.edits import program_subsets

    out = program_subsets(program, max_units=3, max_candidates=12)
    assert out[0].metadata["subset_units"] == [1]
    assert any(item.metadata["subset_units"] == [1, 2] for item in out)
    assert all(len(item.ops) < len(program.ops) for item in out)


def test_split_result_from_dict_roundtrip():
    original = _probe_split("goal_shift", [True, False, True], 0.42)
    restored = SplitResult.from_dict(original.to_dict())
    assert restored.scenario == original.scenario
    assert restored.split == original.split
    assert restored.steps == original.steps
    assert restored.metrics() == original.metrics()
    assert len(restored.episodes) == 3


def test_snapshot_resume_round(tmp_path):
    run_dir = create_run_dir(tmp_path, "run")
    (run_dir / "final").mkdir(exist_ok=True)
    (run_dir / "final" / "metrics.json").write_text("{}")
    (run_dir / "report.md").write_text("# old")
    bundle = tmp_path / "old_bundle.npz"
    bundle.write_bytes(b"old")
    round_dir = snapshot_resume_round(
        run_dir,
        round_index=1,
        bundle_path=str(bundle),
        metadata={"start_cycle": 2},
    )
    assert (round_dir / "final" / "metrics.json").exists()
    assert (round_dir / "report.md").read_text() == "# old"
    assert (round_dir / "bundle.npz").read_bytes() == b"old"
    meta = json.loads((round_dir / "resume.json").read_text())
    assert meta["round"] == 1 and meta["start_cycle"] == 2


def test_resume_requires_final_bundle(tmp_path):
    run_dir = create_run_dir(tmp_path, "run")
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": "run",
            "status": "ok",
            "scenario": {"name": "goal_shift"},
            "bundle": "weights/quad6dof_connectome.npz",
            "cycles": [],
        },
    )
    config = RepairRunConfig(
        out_root=str(tmp_path),
        resume_run="run",
        proposer="template",
    )
    with pytest.raises(ResumeError):
        RepairLoop(config)


def test_artifact_roundtrip(tmp_path):
    b, activity = _activity_and_features()
    run_dir = create_run_dir(tmp_path, "unit-run")
    split = SplitResult(scenario="nominal", split="practice", steps=10, episodes=[_episode("nominal")])
    reg = SplitResult(scenario="nominal", split="regression", steps=10, episodes=[_episode("nominal")])
    write_baseline(
        run_dir,
        scenario=REGISTRY["nominal"],
        practice=split,
        regression=reg,
        activity=activity,
        demo_features=np.zeros((2, 46)),
    )
    record = CycleRecord(
        cycle=1,
        scenario="nominal",
        proposer="template",
        model=None,
        program=EditProgram("p", [], "no-op"),
        validation=ValidationReport(ok=True),
        practice=split.to_dict(),
        regression=reg.to_dict(),
        accepted=False,
        score=1.0,
    )
    write_cycle(run_dir, record)
    write_final(
        run_dir,
        scenario=REGISTRY["nominal"],
        final_practice=split,
        final_regression=reg,
        bundle_path="weights/repair/x.npz",
        report_md="# report\n",
    )
    write_json(run_dir / "manifest.json", {"run_id": "unit-run", "scenario": {"name": "nominal"}})
    data = load_run(tmp_path, "unit-run")
    assert data is not None and len(data["cycles"]) == 1
    cycle = load_cycle(tmp_path, "unit-run", 1)
    assert cycle is not None and cycle["program"]["program_id"] == "p"
