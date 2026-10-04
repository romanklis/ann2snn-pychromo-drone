"""2.5D obstacle geometry: box/cylinder SDF, normals and ray casting."""

from __future__ import annotations

import numpy as np
import pytest

from drone6dof.geometry import BoxObstacle, Cylinder, clearance, ray_cast


def test_box_signed_distance_face_inside_corner():
    b = BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.3))
    assert b.signed_distance((1.0, 0.0)) == pytest.approx(0.5)
    assert b.signed_distance((0.0, 0.0)) == pytest.approx(-0.3)
    assert b.signed_distance((0.5, 0.3)) == pytest.approx(0.0, abs=1e-9)
    assert b.signed_distance((0.9, 0.7)) == pytest.approx(np.hypot(0.4, 0.4), abs=1e-9)


def test_box_closest_point_normal_face_and_corner():
    b = BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.3))
    cp, n, d = b.closest_point_normal((1.0, 0.0))
    assert np.allclose(n, [1.0, 0.0]) and np.allclose(cp, [0.5, 0.0]) and d == pytest.approx(0.5)
    cp, n, d = b.closest_point_normal((0.6, 0.4))
    assert n[0] > 0 and n[1] > 0
    assert np.allclose(cp, [0.5, 0.3]) and d == pytest.approx(np.hypot(0.1, 0.1))


def test_box_rotation_swaps_extents():
    b = BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.3), angle=np.pi / 2.0)
    assert b.signed_distance((0.0, 0.6)) == pytest.approx(0.1, abs=1e-9)
    assert b.signed_distance((0.6, 0.0)) == pytest.approx(0.3, abs=1e-9)


def test_ray_cast_box_and_cylinder():
    box = BoxObstacle(center=(0.0, 0.0), half=(0.5, 0.5))
    angles = np.array([0.0, np.pi])
    r = ray_cast([box], (-2.0, 0.0), angles, 5.0)
    assert r[0] == pytest.approx(1.5)
    assert r[1] == pytest.approx(5.0)  # away from the box -> miss

    cyl = Cylinder(center=(0.0, 0.0), radius=0.5)
    r = ray_cast([cyl], (-2.0, 0.0), np.array([0.0]), 5.0)
    assert r[0] == pytest.approx(1.5)


def test_clearance_is_min_over_obstacles():
    obstacles = [BoxObstacle(center=(0.0, 0.0), half=(0.2, 0.2)),
                 Cylinder(center=(1.0, 0.0), radius=0.2)]
    assert clearance(obstacles, (-0.5, 0.0)) == pytest.approx(0.3)
    assert clearance([], (0.0, 0.0)) == float("inf")
