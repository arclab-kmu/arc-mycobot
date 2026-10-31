"""Hull points behind the lift's table-clearance penalty; CPU only."""

import numpy as np
import pytest

from arc_mycobot.assets.robots.camera_gripper_assets import CAMERA_GRIPPER_PARALLEL_URDF
from arc_mycobot.assets.robots.mycobot_urdf import ARM_JOINTS, LIFT_HOME_POSE, LIFT_START_POSE
from arc_mycobot.kinematics.link_geometry import link_hull_points, stl_vertices
from arc_mycobot.kinematics.urdf_fk import load_chain
from arc_mycobot.tasks.lift.config.mycobot.geometry import CUBE_SIZE, JAW_OFFSET_IN_GRIPPER_BASE

LINKS = ["joint6", "joint6_flange", "gripper_base", "camera_flange"]
MARGIN = 0.005


def lowest_height(link, points, angles):
    chain = load_chain(link, CAMERA_GRIPPER_PARALLEL_URDF)
    position, rotation = chain.fk(angles[: len(chain.actuated)])
    return float((points @ rotation.T + position)[:, 2].min())


def test_extreme_points_keep_the_true_lowest_vertex():
    """400 sampled directions find the lowest hull vertex in any orientation to within 0.5 mm."""
    sample = link_hull_points(["camera_flange"])["camera_flange"]
    dense = link_hull_points(["camera_flange"], directions=20000)["camera_flange"]
    total = len(stl_vertices(CAMERA_GRIPPER_PARALLEL_URDF.parent / "../meshes/camera_flange_hull.stl"))
    assert 10 < len(sample) < total
    rng = np.random.default_rng(0)
    for _ in range(200):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotation *= np.sign(np.linalg.det(rotation))
        assert (sample @ rotation.T)[:, 2].min() == pytest.approx((dense @ rotation.T)[:, 2].min(), abs=0.0005)


@pytest.mark.parametrize("pose", [LIFT_START_POSE, LIFT_HOME_POSE], ids=["start", "grasp_ready"])
def test_lift_poses_are_clear_of_the_table(pose):
    """No clearance penalty at the episode start or at the zero-action target."""
    angles = np.array([pose[name] for name in ARM_JOINTS])
    points = link_hull_points(LINKS)
    for link in LINKS:
        assert lowest_height(link, points[link], angles) > MARGIN, link


def test_gripper_body_hull_stays_out_of_the_cube_between_the_pads():
    """A collider that reached into the jaw would push the cube instead of letting the pads close."""
    body = link_hull_points(["gripper_base"])["gripper_base"]
    assert body[:, 1].max() < JAW_OFFSET_IN_GRIPPER_BASE[1] - CUBE_SIZE / 2


def test_finger_links_fall_back_to_their_visual_hull():
    """No finger collider (see the cube test above), but the clearance term still needs their geometry."""
    points = link_hull_points(["gripper_left2", "gripper_right2"])
    angles = np.array([LIFT_HOME_POSE[name] for name in ARM_JOINTS])
    for link, link_points in points.items():
        assert len(link_points) > 10
        assert lowest_height(link, link_points, angles) > 0.0, link
