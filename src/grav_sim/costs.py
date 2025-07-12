import pickle

import numpy as np
import torch
import trimesh
from alphashape import alphashape
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from scipy.spatial.transform import Rotation


def euclidean_distance(origin, positions, axis=1):
    return np.linalg.norm(positions - origin, axis=axis)


def sum_of_absolute_differences(pose, poses, axis=1):
    return np.sum(np.abs(pose - poses), axis=axis)


def quaternion_distance(pose, poses, degrees=True):
    pose.shape = 1, -1, 3
    joint_count = pose.shape[1]
    poses.shape = -1, joint_count, 3
    batch_size = poses.shape[0]

    all_dists = []
    for batch_i in range(batch_size):
        other_pose = poses[batch_i]
        dists = []
        for joint_i in range(joint_count):
            quat1 = Rotation.from_euler('xyz', pose[0, joint_i], degrees=degrees).as_quat()
            quat2 = Rotation.from_euler('xyz', other_pose[joint_i], degrees=degrees).as_quat()
            quat_diff = Rotation.from_quat(quat1).inv() * Rotation.from_quat(quat2)
            angle = quat_diff.magnitude()
            angle = np.degrees(angle) if degrees else angle
            dists.append(angle)
        all_dists.append(np.sum(dists))
    return np.array(all_dists)


def distance_from_centroid(positions):
    axis = 1 if positions.shape[0] == 3 else 0
    centroid = np.median(positions, axis=axis)
    return np.linalg.norm(centroid - positions, axis=axis - 1)


def distance_from_surface(mesh: trimesh.Trimesh, positions):
    _, distances, _ = trimesh.proximity.closest_point(mesh, positions)
    return distances


def signed_distance_from_surface(mesh: trimesh.Trimesh, positions):
    return trimesh.proximity.signed_distance(mesh, positions)


def distance_from_alpha_shape(point, positions, alpha=.1):
    if len(positions) < 3:
        return np.zeros(len(positions))
    alpha_shape = alphashape(positions, alpha=alpha)
    _, distance, _ = trimesh.proximity.closest_point(alpha_shape, point)
    return distance


def distance_from_convex_hull(point, positions):
    if len(positions) < 3:
        return np.zeros(len(positions))
    convex_hull = trimesh.convex.convex_hull(positions)
    _, distance, _ = trimesh.proximity.closest_point(convex_hull, point)
    return distance


def convex_hull_volume(positions):
    if len(positions) < 3:
        return np.zeros(len(positions))
    return trimesh.convex.convex_hull(positions).volume


def cage_ratio(object_mesh: trimesh.Trimesh, hand_mesh: trimesh.Trimesh):
    object_hull = trimesh.convex.convex_hull(object_mesh)
    body_hull = trimesh.convex.convex_hull(hand_mesh)
    intersection = object_hull.intersection(body_hull)
    if intersection.is_empty:
        return 0
    return intersection.volume / object_hull.volume


_anatomy_loss_fn = AnatomyConstraintLossEE(reduction='none')
_anatomy_loss_fn.setup()


def anatomy_loss(poses):
    pose = torch.FloatTensor(poses)
    return _anatomy_loss_fn(pose).sum(axis=1)


COOR_TO_ANGLES = {
    # [mano index, euler angle index]
    1: [13, 2],
    2: [4, 2],
    3: [15, 2],
    4: [13, 1],
    5: [1, 2],
    6: [2, 2],
    7: [3, 2],
    8: [4, 2],
    9: [5, 2],
    10: [6, 2],
    11: [1, 1],
    12: [10, 2],
    13: [11, 2],
    14: [12, 2],
    15: [4, 1],
    16: [7, 2],
    17: [8, 2],
    18: [9, 2],
    19: [10, 1],
    20: 0,
    21: [0, 2],
    22: [0, 1]
}


def zero_to_one_scale(costs):
    _min = costs.min()
    _max = costs.max()
    return (costs - _min) / (_max - _min)


with open(r'D:\GraVSim\gravsim\assets\lr_model.pkl', 'rb') as file:
    preference_model = pickle.load(file)

with open(r'D:\GraVSim\gravsim\assets\lr_scaler.pkl', 'rb') as file:
    preference_scaler = pickle.load(file)


def preference(joint_position, positions, object_mesh, limb_complement_mesh, poses):
    _angular_distance = euclidean_distance(joint_position, positions)
    _distance_from_centroid = distance_from_centroid(positions)
    _distance_from_object = distance_from_surface(object_mesh, positions)
    _distance_from_body = distance_from_surface(limb_complement_mesh, positions)
    _anatomy_loss = anatomy_loss(poses.reshape(-1, 16, 3))
    # _vemg_abs_sum = vemg_abs_sum(poses)
    _cage_ratio = np.zeros_like(poses)
    _cage_ratio[:] = cage_ratio(object_mesh, limb_complement_mesh)

    shape = len(positions), 10
    X = np.zeros(shape)
    X[:, 3:5] = positions[:, :2]
    X[:, 5:] = np.vstack([
        _angular_distance,
        _distance_from_centroid,
        _distance_from_object,
        # _vemg_abs_sum,
        _cage_ratio]).T

    X[1:, :] = X[1:, :] - X[0, :]
    X = preference_scaler.transform(X)
    return preference_model.predict_proba(X)[:, -1]
