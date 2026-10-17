"""Close looks: the camera is put straight above the target, and a block or knob that isn't quite where the
overview saw it is gripped where it really is."""

import asyncio
import dataclasses

import numpy as np
import pytest

from tests.fake_rig import FakeRig, FakeWorld, FBlock
from tests.synth import Box
from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.look import look_pose, refine_block
from worldarm.act.pick import BLOCKED, HELD, pick
from worldarm.cell.geometry import pose_matrix, rotvec_matrix, vec_of
from worldarm.see.scene import observe


@pytest.mark.parametrize("theta", [0.0, 75.0, -120.0, 180.0])
def test_look_pose_puts_the_target_in_the_middle_of_the_picture(theta):
    """armfarm03's mount: 72 mm off the gripper axis, 87 mm above its frame, tilted ~2 degrees."""
    Rgc = rotvec_matrix(np.array([0.032, 0.01, 0.4]))
    tgc = np.array([68.4, -23.9, -87.0])
    target = np.array([420.0, -310.0, 44.0])
    pose = look_pose(target, theta, (Rgc, tgc), 330.0)
    Rg, tg = pose_matrix({k: getattr(pose, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})
    Rc, tc = Rg @ Rgc, tg + Rg @ tgc
    in_cam = Rc.T @ (target - tc)
    assert np.allclose(in_cam, [0.0, 0.0, 330.0], atol=0.5)
    assert pose.o_z == -1.0 and pose.theta == theta


@pytest.mark.sim
def test_a_block_moved_since_the_overview_is_gripped_where_it_is_now():
    w = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    rig = FakeRig(w)
    sc = observe(asyncio.run(rig.look()), rig.cfg)
    b = sc.table_blocks[0]
    shift = vec_of(35.0) * 25.0                     # along its short side: a finger would land on it
    w.blocks[0].x += shift[0]
    w.blocks[0].y += shift[1]
    first = plan_grasp(b, rig.cfg, sc.table_blocks)
    nb, neighbors, info = asyncio.run(refine_block(rig, b, sc.table_blocks, (first.theta, first.alt_theta)))
    assert info["status"] == "refined" and abs(info["moved_mm"] - 25.0) < 4.0
    assert np.linalg.norm(nb.center[:2] - [w.blocks[0].x, w.blocks[0].y]) < 4.0
    r = asyncio.run(pick(rig, plan_grasp(nb, rig.cfg, neighbors)))
    assert r.status == HELD and rig.held is w.blocks[0]


@pytest.mark.sim
def test_without_the_close_look_the_same_move_lands_a_finger_on_it():
    """The control for the test above: going by the overview alone, the finger lands on the block."""
    w = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    rig = FakeRig(w)
    rig.cfg.camera.look_mm = 0.0
    sc = observe(asyncio.run(rig.look()), rig.cfg)
    b = sc.table_blocks[0]
    shift = vec_of(35.0) * 25.0
    w.blocks[0].x += shift[0]
    w.blocks[0].y += shift[1]
    first = plan_grasp(b, rig.cfg, sc.table_blocks)
    nb, neighbors, info = asyncio.run(refine_block(rig, b, sc.table_blocks, (first.theta, first.alt_theta)))
    assert info["status"] == "no look" and nb is b
    r = asyncio.run(pick(rig, plan_grasp(nb, rig.cfg, neighbors)))
    assert r.status == BLOCKED and rig.held is None


@pytest.mark.sim
def test_a_block_gone_from_where_it_was_falls_back_to_the_overview():
    w = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    rig = FakeRig(w)
    sc = observe(asyncio.run(rig.look()), rig.cfg)
    b = sc.table_blocks[0]
    w.blocks.clear()
    nb, _, info = asyncio.run(refine_block(rig, b, sc.table_blocks, (0.0, 180.0), say=lambda *_: None))
    assert info["status"] == "not found" and nb is b


@pytest.mark.sim
def test_close_looks_are_saved_when_asked(tmp_path):
    w = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    rig = FakeRig(w)
    rig.look_dir = tmp_path / "looks"
    sc = observe(asyncio.run(rig.look()), rig.cfg)
    b = sc.table_blocks[0]
    asyncio.run(refine_block(rig, b, sc.table_blocks, (0.0, 180.0), say=lambda *_: None))
    assert len(list((tmp_path / "looks").glob("*.jpg"))) == 1 and len(list((tmp_path / "looks").glob("*.npz"))) == 1
