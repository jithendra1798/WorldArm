"""Forward kinematics and ball velocity, checked against arms with known answers."""

import math

import numpy as np
import pytest

from worldarm.toss.kinematics import Chain

# A planar 3-link arm in the x-z plane (all axes along y), like the xArm's shoulder/elbow/wrist pitch,
# with a base yaw joint first: links 400, 300, 150 mm.
PLANAR = {
    "name": "test", "kinematic_param_type": "SVA",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0}},
        {"id": "l0", "parent": "yaw", "translation": {"x": 0, "y": 0, "z": 250}},
        {"id": "l1", "parent": "j1", "translation": {"x": 400, "y": 0, "z": 0}},
        {"id": "l2", "parent": "j2", "translation": {"x": 300, "y": 0, "z": 0}},
        {"id": "l3", "parent": "j3", "translation": {"x": 150, "y": 0, "z": 0}},
    ],
    "joints": [
        {"id": "yaw", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": -360, "max": 360},
        {"id": "j1", "type": "revolute", "parent": "l0", "axis": {"x": 0, "y": -1, "z": 0}, "min": -120, "max": 120},
        {"id": "j2", "type": "revolute", "parent": "l1", "axis": {"x": 0, "y": -1, "z": 0}, "min": -170, "max": 170},
        {"id": "j3", "type": "revolute", "parent": "l2", "axis": {"x": 0, "y": -1, "z": 0}, "min": -170, "max": 170},
    ],
}


def planar_tip(q):
    """Analytic: yaw q0, then pitch angles q1..q3 (positive = upward, axis -y) in the vertical plane."""
    a1 = math.radians(q[1])
    a2 = a1 + math.radians(q[2])
    a3 = a2 + math.radians(q[3])
    r = 400 * math.cos(a1) + 300 * math.cos(a2) + 150 * math.cos(a3)
    z = 250 + 400 * math.sin(a1) + 300 * math.sin(a2) + 150 * math.sin(a3)
    y0 = math.radians(q[0])
    return np.array([r * math.cos(y0), r * math.sin(y0), z])


@pytest.mark.parametrize("q", [(0, 0, 0, 0), (30, 20, -40, 10), (-90, -30, 60, 45)])
def test_forward_kinematics_matches_the_analytic_arm(q):
    c = Chain.from_sva(PLANAR)
    assert np.allclose(c.point(q), planar_tip(q), atol=1e-6)


def test_ball_velocity_from_joint_speeds_matches_finite_differences():
    c = Chain.from_sva(PLANAR)
    q, qd = np.array([10.0, 20.0, -30.0, 15.0]), np.array([0.0, 180.0, 180.0, 180.0])
    dt = 1e-6
    numeric = (c.point(q + qd * dt) - c.point(q - qd * dt)) / (2 * dt)
    assert np.allclose(c.velocity(q, qd), numeric, rtol=1e-5, atol=1e-3)


def test_parallel_joints_add_their_speeds_at_the_tip():
    """Stretched out, three pitch joints at 180 deg/s move the tip at pi * (850 + 450 + 150) mm/s."""
    c = Chain.from_sva(PLANAR)
    v = c.velocity((0, 0, 0, 0), (0, 180, 180, 180))
    assert np.linalg.norm(v) == pytest.approx(math.pi * (850 + 450 + 150), rel=1e-6)
    single = c.velocity((0, 0, 0, 0), (0, 180, 0, 0))
    assert np.linalg.norm(single) == pytest.approx(math.pi * 850, rel=1e-6)


def test_urdf_and_sva_agree():
    urdf = """<robot name="t">
      <link name="base"/><link name="a"/><link name="b"/><link name="c"/><link name="d"/>
      <joint name="yaw" type="revolute"><parent link="base"/><child link="a"/>
        <origin xyz="0 0 0.25" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28"/></joint>
      <joint name="j1" type="revolute"><parent link="a"/><child link="b"/>
        <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 -1 0"/><limit lower="-2" upper="2"/></joint>
      <joint name="j2" type="revolute"><parent link="b"/><child link="c"/>
        <origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 -1 0"/><limit lower="-3" upper="3"/></joint>
      <joint name="j3" type="revolute"><parent link="c"/><child link="d"/>
        <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 -1 0"/><limit lower="-3" upper="3"/></joint>
    </robot>"""
    u = Chain.from_urdf(urdf)
    q = (25, 15, -35, 20)
    tool = (150.0, 0.0, 0.0)
    assert np.allclose(u.point(q, tool), planar_tip(q), atol=1e-6)


def test_urdf_with_a_dangling_world_link_first_still_reads_the_whole_arm():
    """armfarm05's own URDF lists an unconnected <link name="world"/> before the arm's base."""
    urdf = """<robot name="r"><link name="world"/><link name="base"/><link name="l1"/><link name="l2"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.3"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.4 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-2" upper="2"/></joint></robot>"""
    chain = Chain.from_urdf(urdf)
    assert chain.n == 2 and np.allclose(chain.point([0.0, 0.0]), (400.0, 0.0, 300.0))
