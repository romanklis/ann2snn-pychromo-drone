"""LiDAR scan model and sensor cues."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.geometry import BoxObstacle
from drone6dof.sensor import SENSOR_CUE_DIM, SensorConfig, scan, sensor_features


def test_scan_hits_nearest_surface():
    cfg = SensorConfig(k=8, range_sigma=0.0, dropout_p=0.0, max_range_miss_p=0.0)
    obstacles = (BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.5)),)
    ranges = scan(obstacles, (-2.0, 0.0), cfg)
    assert ranges.shape == (8,)
    assert ranges[0] == pytest.approx(1.5)          # +x beam hits the box
    assert np.all(ranges[1:] == cfg.r_max)          # other beams miss


def test_scan_noise_and_dropout_stay_in_bounds():
    cfg = SensorConfig(k=32, range_sigma=0.05, dropout_p=0.5,
                       max_range_miss_p=0.5, seed=1)
    rng = np.random.default_rng(0)
    clean = scan((), (0.0, 0.0),
                 SensorConfig(k=32, range_sigma=0.0, dropout_p=0.0, max_range_miss_p=0.0))
    assert np.all(clean == cfg.r_max)                       # no obstacles -> r_max
    ranges = scan((), (0.0, 0.0), cfg, rng)                 # noise still bounded
    assert ranges.min() >= 0.0 and ranges.max() <= cfg.r_max
    obstacles = (BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.5)),)
    ranges = scan(obstacles, (-2.0, 0.0), cfg, rng)
    assert ranges.min() >= 0.0 and ranges.max() <= cfg.r_max


def test_sensor_features_shape_and_cues():
    cfg = SensorConfig(k=16, range_sigma=0.0, dropout_p=0.0, max_range_miss_p=0.0)
    obstacles = (BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.5)),)
    feats = sensor_features((-2.0, 0.0), obstacles, cfg)
    assert feats.shape == (16 + SENSOR_CUE_DIM,)
    ranges = feats[:16]
    assert np.all(ranges >= 0.0) and np.all(ranges <= 1.0)
    min_range, bx, by = feats[16], feats[17], feats[18]
    assert min_range == pytest.approx(1.5 / cfg.r_max)
    assert np.isclose(np.hypot(bx, by), 1.0)        # bearing is a unit vector
    assert bx == pytest.approx(1.0) and by == pytest.approx(0.0)
