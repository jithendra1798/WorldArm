import pytest
from viam.proto.common import Pose

from worldarm.safety import Limits, SafetyError, check_gripper_target

LIMITS = Limits(table_z_mm=2.0, floor_margin_mm=5.0, fingertip_mm=60.0)


def test_pointing_down_with_fingertips_above_the_floor_is_allowed():
    check_gripper_target(Pose(x=400, y=-200, z=100, o_z=-1), LIMITS)  # fingertips at z = 40


def test_fingertips_below_the_floor_are_rejected():
    with pytest.raises(SafetyError, match="fingertips z=0"):
        check_gripper_target(Pose(x=400, y=-200, z=60, o_z=-1), LIMITS)


def test_target_past_the_front_wall_is_rejected():
    with pytest.raises(SafetyError, match="x=700"):
        check_gripper_target(Pose(x=700, y=0, z=300, o_z=-1), LIMITS)


def test_horizontal_gripper_fingertips_reach_forward():
    with pytest.raises(SafetyError, match="fingertips x=660"):
        check_gripper_target(Pose(x=600, y=0, z=200, o_x=1), LIMITS)
