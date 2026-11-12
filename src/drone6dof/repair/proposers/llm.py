"""Optional OpenAI-compatible LLM proposer with budgeted tools."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..types import EditProgram
from .base import ProposalContext, Proposer
from .llm_tools import TOOL_SCHEMA, RepairToolbox

__all__ = ["LLMConfig", "OpenAIClient", "LLMProposer", "extract_json"]


def _env(name: str, default: str = "") -> str:
    """Read an env var, treating unset and empty values the same."""
    value = os.environ.get(name)
    return default if value is None or str(value).strip() == "" else str(value)


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    supports_tools: bool = True
    supports_sampling: bool = True
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "LLMConfig":
        """Resolve provider-specific defaults.

        Precedence for the provider is ``provider`` > ``REPAIR_LLM_PROVIDER`` >
        ``openai``.  DeepSeek-specific variables win over the generic
        ``REPAIR_LLM_*`` ones so a DeepSeek setup is self-contained.
        """
        provider_name = str(
            provider or _env("REPAIR_LLM_PROVIDER", "openai")
        ).strip().lower().replace("_", "-")
        if provider_name == "deepseek":
            base_url = (
                _env("DEEPSEEK_BASE_URL")
                or _env("REPAIR_LLM_BASE_URL")
                or "https://api.deepseek.com"
            )
            api_key = _env("DEEPSEEK_API_KEY") or _env("REPAIR_LLM_API_KEY")
            model_name = (
                model
                or _env("DEEPSEEK_MODEL")
                or _env("REPAIR_LLM_MODEL")
                or "deepseek-chat"
            )
        else:
            base_url = (
                _env("REPAIR_LLM_BASE_URL")
                or "https://api.openai.com/v1"
            )
            api_key = _env("REPAIR_LLM_API_KEY")
            model_name = (
                model
                or _env("REPAIR_LLM_MODEL")
                or "gpt-4o-mini"
            )
        # DeepSeek's reasoner currently has no function-calling and ignores
        # sampling knobs.  Treat that as a model capability, not a provider one.
        reasoner = "reasoner" in str(model_name).lower()
        timeout_env = (
            "DEEPSEEK_TIMEOUT" if provider_name == "deepseek" else "REPAIR_LLM_TIMEOUT"
        )
        timeout = float(_env(timeout_env, _env("REPAIR_LLM_TIMEOUT", "120")))
        default_max_tokens = "8192" if reasoner else "4096"
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            provider=provider_name,
            supports_tools=not reasoner,
            supports_sampling=not reasoner,
            temperature=float(_env("REPAIR_LLM_TEMPERATURE", "0.2")),
            max_tokens=int(_env("REPAIR_LLM_MAX_TOKENS", default_max_tokens)),
            timeout=timeout,
        )


class OpenAIClient:
    """Minimal JSON chat-completions client using only the standard library.

    The wire format is OpenAI-compatible, which covers OpenAI, DeepSeek,
    vLLM/Ollama gateways, and other compatible endpoints.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def chat_completions_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop provider-specific response fields before a follow-up request.

        DeepSeek's reasoner returns ``reasoning_content``; echoing it back can be
        rejected or waste context.  Only fields accepted by chat-completions are
        forwarded.
        """
        allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
        clean = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            clean.append({key: value for key, value in message.items() if key in allowed})
        return clean

    def build_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": self._sanitize_messages(messages),
            "max_tokens": int(self.config.max_tokens),
        }
        if self.config.supports_sampling:
            payload["temperature"] = float(self.config.temperature)
        if tools and self.config.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self.config.api_key:
            keys = (
                "DEEPSEEK_API_KEY or REPAIR_LLM_API_KEY"
                if self.config.provider == "deepseek"
                else "REPAIR_LLM_API_KEY"
            )
            raise RuntimeError(
                f"no API key for provider {self.config.provider!r}: set {keys}; "
                "use --proposer template to run without an LLM"
            )
        url = self.chat_completions_url()
        payload = self.build_payload(messages, tools=tools)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        data = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=float(self.config.timeout)) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if attempt == 0 and (exc.code == 429 or 500 <= exc.code < 600):
                    time.sleep(1.0)
                    continue
                raise RuntimeError(
                    f"{self.config.provider} LLM HTTP {exc.code} "
                    f"(model {self.config.model}): {body[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                raise RuntimeError(
                    f"{self.config.provider} LLM request failed "
                    f"(model {self.config.model}): {exc}"
                ) from exc
        if data is None:
            raise RuntimeError(
                f"{self.config.provider} LLM request failed without a response"
            )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {data}")
        return choices[0].get("message", {})


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM reply."""
    if not isinstance(text, str):
        raise ValueError("LLM content is not text")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = min(
        [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0],
        default=-1,
    )
    if start < 0:
        raise ValueError("no JSON object found in the LLM reply")
    for end in range(len(candidate), start, -1):
        try:
            return json.loads(candidate[start:end])
        except json.JSONDecodeError:
            continue
    raise ValueError("could not parse the JSON object in the LLM reply")


def _normalize_ops(data: Any) -> Dict[str, Any]:
    if isinstance(data, list):
        data = {"ops": data, "rationale": ""}
    if not isinstance(data, dict):
        raise ValueError("program must be a JSON object")
    if "program" in data and isinstance(data["program"], dict):
        data = data["program"]
    ops = data.get("ops", [])
    if not isinstance(ops, list):
        raise ValueError("program.ops must be a list")
    normalized = []
    for op in ops:
        if not isinstance(op, dict):
            raise ValueError("each op must be an object")
        op = dict(op)
        if "kind" not in op:
            for alias in ("type", "op", "action"):
                if alias in op:
                    op["kind"] = op.pop(alias)
                    break
        normalized.append(op)
    return {
        "program_id": str(data.get("program_id", "llm-program")),
        "ops": normalized,
        "rationale": str(data.get("rationale", "")),
        "proposer": "llm",
        "metadata": dict(data.get("metadata", {})),
    }


class LLMProposer(Proposer):
    name = "llm"

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        *,
        client=None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.config = config or LLMConfig.from_env(provider=provider, model=model)
        self.client = client or OpenAIClient(self.config)
        self.transcript: List[Dict[str, Any]] = []
        self.last_toolbox: Optional[RepairToolbox] = None

    def _system_prompt(self) -> str:
        tool_note = (
            "You can inspect spare units, probe their rate response, and request "
            "a bounded practice rollout through the provided tools. "
            if self.config.supports_tools
            else "Tool calling is unavailable for this model. Put the final JSON "
            "edit program in the message content; do not emit tool calls and do "
            "not put the final JSON in reasoning_content. Keep internal reasoning "
            "brief and answer immediately. "
        )
        return (
            "You repair a spiking neural network policy by writing small programs "
            "into neurons that are unused (the 'white space'). You may not use "
            "gradients, optimizers, weight fitting, new demonstrations, or the "
            "simulator directly. "
            + tool_note
            + "Every edit must target spare units, must be quiet on the in-distribution "
            "demos, and should switch on only in the failure context. The SNN is "
            "rate-coded: a unit's contribution is w_out[:, unit] * spike rate; w_in "
            "selects features; bias shifts the threshold; add_edge can make a gate "
            "drive a relay. Prefer a few surgical units over a broad rewrite.\n\n"
            "Edit kinds: set_input_row (unit, values[46]); add_input_row (unit, "
            "values[46]); set_output_col (unit, values[3]); add_output_col (unit, "
            "values[3]); set_bias (unit, value); add_bias (unit, value); add_edge "
            "(src, dst, magnitude); set_edge (src, dst, magnitude); silence_output "
            "(unit); isolate_unit (unit, zero recurrent synapses touching the unit). "
            "Before submitting, call validate_program. Prefer a smaller validated "
            "program over adding an untested unit. Use a few episodes per "
            "run_practice and keep budget for the final candidate. "
            "Return JSON: {\"ops\": [...], \"rationale\": \"...\"}."
        )

    def _user_prompt(self, context: ProposalContext) -> str:
        summary = context.activity.summary
        compact = not self.config.supports_tools
        spare = summary.get("top_spare", [])[: 8 if compact else 20]
        history_limit = 3 if compact else 6
        history = [
            {
                "cycle": h.get("cycle"),
                "accepted": h.get("accepted"),
                "program": (h.get("program") or {}).get("rationale", ""),
                "score": h.get("score"),
                "success": (h.get("practice") or {}).get("metrics", {}).get("success_rate"),
                "progress": (h.get("practice") or {}).get("metrics", {}).get("mean_final_progress"),
                "validation": {
                    "ok": (h.get("validation") or {}).get("ok"),
                    "issues": (h.get("validation") or {}).get("issues", [])[:4],
                    "warnings": (h.get("validation") or {}).get("warnings", [])[:4],
                    "quietness": (h.get("validation") or {}).get("quietness", {}),
                    "auto_isolated_units": (h.get("validation") or {}).get("auto_isolated_units", []),
                    "pruned_units": (h.get("validation") or {}).get("pruned_units", []),
                },
            }
            for h in context.history[-history_limit:]
        ]
        examples = context.diagnosis.examples[: 2 if compact else 4]
        demo = context.demo_features
        target = context.target_features
        names = summary.get("feature_names") or []
        feature_stats = []
        if getattr(demo, "size", 0) and getattr(target, "size", 0) and len(names):
            import numpy as np

            d_mean = np.mean(demo, axis=0)
            d_std = np.std(demo, axis=0)
            t_mean = np.mean(target, axis=0)
            t_std = np.std(target, axis=0)
            keep = set(range(min(9, len(names)))) | set(range(41, min(46, len(names))))
            if compact:
                ranges = [
                    (abs(float(t_mean[i] - d_mean[i])) / float(d_std[i] + 1e-6), i)
                    for i in range(9, min(41, len(names)))
                ]
                keep |= {i for _, i in sorted(ranges, reverse=True)[:5]}
            for i, name in enumerate(names):
                if i >= demo.shape[1] or i not in keep:
                    continue
                feature_stats.append(
                    {
                        "index": i,
                        "name": name,
                        "demo_mean": round(float(d_mean[i]), 4),
                        "demo_std": round(float(d_std[i]), 4),
                        "target_mean": round(float(t_mean[i]), 4),
                        "target_std": round(float(t_std[i]), 4),
                    }
                )
        payload = {
            "cycle": context.cycle,
            "scenario": context.scenario.name,
            "scenario_description": context.scenario.description,
            "goal": list(context.scenario.scene.goal),
            "features": summary.get("feature_names"),
            "feature_stats": feature_stats,
            "actions": summary.get("action_names"),
            "activity": {
                "silent_frac": summary.get("silent_frac"),
                "rare_frac": summary.get("rare_frac"),
                "spare_count": summary.get("spare_count"),
                "critical_count": summary.get("critical_count"),
                "top_spare": spare,
            },
            "diagnosis": {
                "failure_counts": context.diagnosis.failure_counts,
                "metrics": context.diagnosis.split_metrics,
                "examples": examples,
                "notes": context.diagnosis.notes,
            },
            "best_metrics": context.best_metrics,
            "history": history,
            "budget": context.budget.to_dict(),
        }
        if compact:
            payload["answer_style"] = (
                "Return one JSON edit program immediately. Keep internal reasoning "
                "brief; the JSON in content is the only thing parsed."
            )
        return (
            "Repair this SNN on the shifted scenario. Diagnose the failure, choose "
            "spare units, probe if useful, then submit a small edit program.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )

    def propose(self, context: ProposalContext) -> EditProgram:
        toolbox = RepairToolbox(
            context.base_bundle,
            context.activity,
            budget=context.budget,
            demo_features=context.demo_features,
            target_features=context.target_features,
            practice_callback=context.practice_callback,
            candidate_callback=getattr(context, "candidate_callback", None),
            spare_units=[
                int(item["unit"])
                for item in context.activity.summary.get("top_spare", [])
            ],
            allow_live=bool(getattr(context, "allow_live", False)),
        )
        self.last_toolbox = toolbox
        self.transcript = []
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(context)},
        ]
        last_error = ""
        tools = TOOL_SCHEMA if self.config.supports_tools else None
        max_turns = max(1, int(context.budget.max_turns))
        budget_nudged = False
        final_nudged = False
        for turn in range(max_turns):
            if (
                not budget_nudged
                and toolbox.practice_runs >= int(context.budget.max_practice_rollouts)
            ):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Practice budget is exhausted. Call submit_program now "
                            "with your best validated candidate; do not start another "
                            "exploration."
                        ),
                    }
                )
                budget_nudged = True
            if not final_nudged and turn == max_turns - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "This is the final turn. Submit the best validated program "
                            "now with submit_program, or return the JSON program in "
                            "content."
                        ),
                    }
                )
                final_nudged = True
            message = self.client.complete(messages, tools=tools)
            self.transcript.append({"role": "assistant", "message": message})
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                messages.append(message)
                for call in tool_calls:
                    fn = call.get("function", {}) if isinstance(call, dict) else {}
                    name = str(fn.get("name", ""))
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except json.JSONDecodeError as exc:
                        args = {}
                        last_error = f"bad tool arguments: {exc}"
                    result = toolbox.call(name, args)
                    self.transcript.append(
                        {"role": "tool", "name": name, "arguments": args, "result": result}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id", name)),
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
                    if name == "submit_program" and result.get("ok") and toolbox.submitted:
                        return self._finalize(toolbox.submitted)
                continue
            content = message.get("content") or ""
            if not content.strip():
                messages.append(
                    {"role": "user", "content": "Return a JSON edit program."}
                )
                continue
            try:
                data = _normalize_ops(extract_json(content))
                program = EditProgram.from_dict(data)
                return self._finalize(program)
            except Exception as exc:
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Invalid program: {exc}. Return only a valid JSON object "
                            "with an 'ops' list and a 'rationale'."
                        ),
                    }
                )
        if toolbox.submitted is not None:
            return self._finalize(toolbox.submitted)
        if toolbox.last_validated is not None:
            program = toolbox.last_validated
            program.metadata.setdefault("fallback", "last_validated_program")
            program.metadata.setdefault(
                "quietness", toolbox.last_validated_quietness
            )
            return self._finalize(program)
        raise RuntimeError(f"LLM proposer produced no valid program ({last_error})")

    def _finalize(self, program: EditProgram) -> EditProgram:
        program.proposer = "llm"
        program.metadata.setdefault("provider", self.config.provider)
        program.metadata.setdefault("model", self.config.model)
        program.metadata.setdefault("turns", len(self.transcript))
        program.metadata.setdefault("tool_calls", 0 if self.last_toolbox is None else self.last_toolbox.calls)
        return program
