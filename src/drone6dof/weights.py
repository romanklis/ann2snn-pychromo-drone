"""Weight bundles for the connectome ANN / SNN.

A bundle is a self-contained ``.npz`` (topology + trained weights + a config
fingerprint) so inference needs neither torch nor the training code.  The
fingerprint is a hash of the behavioural configuration (plant, scene, teacher,
limits, dimensions); loading a bundle trained for a different configuration is
refused rather than silently serving a stale policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import (
    CONTROL_LIMIT,
    DS_TEACHER_KWARGS,
    ESTIMATOR,
    INIT_STATE,
    OBSTACLE_LAYOUT,
    QUAD_SCENE,
    SENSOR,
    SENSOR_SUITE,
)
from .params import DT, QuadParams
from .field import FieldConfig
from .policy import policy_input_dim
from .sensor import SensorConfig
from .sensors import SensorConfig as SensorSuiteConfig

__all__ = [
    "FORMAT",
    "FIELD_FORMAT",
    "WeightsMissing",
    "WeightsMismatch",
    "fingerprint",
    "make_fields",
    "default_fields",
    "make_field_fields",
    "default_field_fields",
    "save_weights",
    "load_weights",
    "save_field_weights",
    "load_field_weights",
    "DEFAULT_WEIGHTS_PATH",
    "DEFAULT_REF_IO_PATH",
    "DEFAULT_FIELD_PATH",
]

FORMAT = "ann2snn.drone6dof.connectome@3"
FIELD_FORMAT = "ann2snn.drone6dof.field@1"

#: repo-root ``weights/`` (src/drone6dof/weights.py -> parents[2])
_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_PATH = _WEIGHTS_DIR / "quad6dof_connectome.npz"
DEFAULT_REF_IO_PATH = _WEIGHTS_DIR / "quad6dof_reference_io.npz"
DEFAULT_FIELD_PATH = _WEIGHTS_DIR / "quad6dof_field.npz"


class WeightsMissing(FileNotFoundError):
    """No bundle at the requested path; run ``make train`` to create one."""


class WeightsMismatch(ValueError):
    """The bundle was trained for a different configuration (fingerprint mismatch)."""


def fingerprint(fields: Dict) -> str:
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_fields(
    *,
    scene,
    params: Optional[QuadParams] = None,
    teacher_kwargs: Optional[Dict] = None,
    control_limit: float = CONTROL_LIMIT,
    init_state=INIT_STATE,
    connectome_steps: int = 3,
    micro_steps: int = 10,
    v_th: float = 1.0,
    seed: int = 42,
    state_source: str = "true",
    sensor: Optional[SensorConfig] = None,
    layout: Optional[Dict] = None,
    sensor_suite: Optional[SensorSuiteConfig] = None,
    estimator: Optional[Dict] = None,
) -> Dict:
    """Canonical behavioural config the bundle is valid for."""
    params = params or QuadParams()
    teacher_kwargs = teacher_kwargs if teacher_kwargs is not None else DS_TEACHER_KWARGS
    sensor = sensor or SENSOR
    layout = layout if layout is not None else OBSTACLE_LAYOUT
    sensor_suite = sensor_suite or SENSOR_SUITE
    estimator = estimator if estimator is not None else ESTIMATOR
    return {
        "format": FORMAT,
        "pos_dim": 3,
        "n_in": policy_input_dim(sensor),
        "n_out": 3,
        "control_limit": float(control_limit),
        "init_state": [float(v) for v in init_state],
        "scene": scene.to_dict(),
        "teacher_kwargs": {k: float(v) for k, v in sorted(teacher_kwargs.items())},
        "connectome_steps": int(connectome_steps),
        "micro_steps": int(micro_steps),
        "v_th": float(v_th),
        "inner_hz": float(params.inner_hz),
        "state_source": state_source,
        "sensor": sensor.to_dict(),
        "layout": dict(layout),
        "sensor_suite": sensor_suite.to_dict(),
        "estimator": dict(estimator),
    }


def default_fields(**overrides) -> Dict:
    """Fields for the ``quad6dof`` example as shipped."""
    base = dict(
        scene=QUAD_SCENE.instantiate(0),
        params=QuadParams(),
        teacher_kwargs=DS_TEACHER_KWARGS,
        control_limit=CONTROL_LIMIT,
        init_state=INIT_STATE,
        connectome_steps=3,
        micro_steps=10,
        v_th=1.0,
        seed=42,
        state_source="estimate",
        sensor=SENSOR,
        layout=OBSTACLE_LAYOUT,
        sensor_suite=SENSOR_SUITE,
        estimator=ESTIMATOR,
    )
    base.update(overrides)
    return make_fields(**base)


def save_weights(path, *, w_in, w_out, w_mag, edges, polarity, fields: Dict, **scalars) -> str:
    """Write a bundle to ``path`` (``.npz``). Returns the path as ``str``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w_in": np.asarray(w_in, dtype=np.float64),
        "w_out": np.asarray(w_out, dtype=np.float64),
        "w_mag": np.asarray(w_mag, dtype=np.float64),
        "edges": np.asarray(edges, dtype=np.int32),
        "polarity": np.asarray(polarity, dtype=np.float64),
        "fingerprint": np.array(fingerprint(fields)),
        "fields_json": np.array(json.dumps(fields, sort_keys=True)),
    }
    for key, value in scalars.items():
        payload[key] = np.array(value)
    np.savez(path, **payload)
    return str(path)


def make_field_fields(
    *,
    scene,
    field_config: Optional[FieldConfig] = None,
    params: Optional[QuadParams] = None,
    sensor: Optional[SensorConfig] = None,
    layout: Optional[Dict] = None,
    sensor_suite: Optional[SensorSuiteConfig] = None,
    estimator: Optional[Dict] = None,
    connectome_steps: int = 3,
    micro_steps: int = 10,
    v_th: float = 1.0,
    seed: int = 42,
    state_source: str = "estimate",
    n_neurons: int = 128,
    n_in: Optional[int] = None,
) -> Dict:
    field_config = field_config or FieldConfig()
    params = params or QuadParams()
    sensor = sensor or SENSOR
    layout = layout if layout is not None else OBSTACLE_LAYOUT
    sensor_suite = sensor_suite or SENSOR_SUITE
    estimator = estimator if estimator is not None else ESTIMATOR
    return {
        "format": FIELD_FORMAT,
        "pos_dim": 3,
        "n_in": int(n_in if n_in is not None else policy_input_dim(sensor)),
        "n_out": field_config.k,
        "n_neurons": int(n_neurons),
        "field": field_config.to_dict(),
        "scene": scene.to_dict(),
        "init_state": [float(v) for v in INIT_STATE],
        "control_limit": float(CONTROL_LIMIT),
        "connectome_steps": int(connectome_steps),
        "micro_steps": int(micro_steps),
        "v_th": float(v_th),
        "seed": int(seed),
        "state_source": state_source,
        "sensor": sensor.to_dict(),
        "layout": dict(layout),
        "sensor_suite": sensor_suite.to_dict(),
        "estimator": dict(estimator),
        "inner_hz": float(params.inner_hz),
    }


def default_field_fields(**overrides) -> Dict:
    base = dict(
        scene=QUAD_SCENE.instantiate(0),
        field_config=FieldConfig(),
        params=QuadParams(),
        sensor=SENSOR,
        layout=OBSTACLE_LAYOUT,
        sensor_suite=SENSOR_SUITE,
        estimator=ESTIMATOR,
        seed=42,
        state_source="estimate",
    )
    base.update(overrides)
    return make_field_fields(**base)


def save_field_weights(path, *, w_in, w_out, w_mag, edges, polarity, fields: Dict, **scalars) -> str:
    """Field bundle (same npz schema as the connectome bundle)."""
    return save_weights(path, w_in=w_in, w_out=w_out, w_mag=w_mag, edges=edges,
                        polarity=polarity, fields=fields, **scalars)


def load_field_weights(path=DEFAULT_FIELD_PATH, *, expect_fields: Optional[Dict] = None) -> Dict:
    """Load a field bundle (format ``FIELD_FORMAT``)."""
    path = Path(path)
    if not path.exists():
        raise WeightsMissing(
            f"no field bundle at {path}; run `make train-field` to distil and export one"
        )
    with np.load(path, allow_pickle=False) as data:
        fields = json.loads(str(data["fields_json"]))
        if fields.get("format") != FIELD_FORMAT:
            raise WeightsMismatch(
                f"unsupported field bundle format {fields.get('format')!r} "
                f"(expected {FIELD_FORMAT!r})"
            )
        w_in = np.array(data["w_in"], dtype=np.float64)
        w_out = np.array(data["w_out"], dtype=np.float64)
        edges = np.array(data["edges"], dtype=np.int32)
        bundle = {
            "w_in": w_in,
            "w_out": w_out,
            "w_mag": np.array(data["w_mag"], dtype=np.float64),
            "edges": edges,
            "polarity": np.array(data["polarity"], dtype=np.float64),
            "fingerprint": str(data["fingerprint"]),
            "fields": fields,
            "field": fields.get("field", {}),
        }
        if "seed" in data.files:
            bundle["seed"] = int(data["seed"])
    n_neurons = int(w_out.shape[1])
    bundle.update({
        "n_neurons": n_neurons,
        "n_in": int(w_in.shape[1]),
        "n_out": int(w_out.shape[0]),
        "k": int(edges.shape[1] // max(n_neurons, 1)),
        "connectome_steps": int(fields.get("connectome_steps", 3)),
        "micro_steps": int(fields.get("micro_steps", 10)),
        "v_th": float(fields.get("v_th", 1.0)),
    })
    if expect_fields is not None:
        if bundle["fingerprint"] != fingerprint(expect_fields):
            raise WeightsMismatch(
                "field bundle fingerprint does not match this configuration; run "
                "`make train-field` to regenerate it"
            )
    return bundle


def load_weights(path=DEFAULT_WEIGHTS_PATH, *, expect_fields: Optional[Dict] = None) -> Dict:
    """Load a bundle, optionally validating it against ``expect_fields``."""
    path = Path(path)
    if not path.exists():
        raise WeightsMissing(
            f"no weight bundle at {path}; run `make train` to distil and export one "
            "(the committed bundle should exist in weights/)"
        )
    with np.load(path, allow_pickle=False) as data:
        fields = json.loads(str(data["fields_json"]))
        got_format = fields.get("format")
        if got_format != FORMAT:
            raise WeightsMismatch(
                f"unsupported weight bundle format {got_format!r} (expected {FORMAT!r})"
            )
        w_in = np.array(data["w_in"], dtype=np.float64)
        w_out = np.array(data["w_out"], dtype=np.float64)
        edges = np.array(data["edges"], dtype=np.int32)
        seed = int(data["seed"]) if "seed" in data.files else 42
        bundle = {
            "w_in": w_in,
            "w_out": w_out,
            "w_mag": np.array(data["w_mag"], dtype=np.float64),
            "edges": edges,
            "polarity": np.array(data["polarity"], dtype=np.float64),
            "fingerprint": str(data["fingerprint"]),
            "fields": fields,
        }
    # dimensions/scalars: arrays are authoritative, the rest come from the fields
    n_neurons = int(w_out.shape[1])
    bundle.update(
        {
            "n_neurons": n_neurons,
            "n_in": int(w_in.shape[1]),
            "n_out": int(w_out.shape[0]),
            "k": int(edges.shape[1] // max(n_neurons, 1)),
            "seed": seed,
            "connectome_steps": int(fields.get("connectome_steps", 3)),
            "micro_steps": int(fields.get("micro_steps", 10)),
            "v_th": float(fields.get("v_th", 1.0)),
        }
    )
    if expect_fields is not None:
        expected = fingerprint(expect_fields)
        if bundle["fingerprint"] != expected:
            raise WeightsMismatch(
                "weight bundle fingerprint does not match this configuration; run "
                "`make train` to regenerate it"
            )
    return bundle
