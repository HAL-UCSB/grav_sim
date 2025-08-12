import os
import pathlib

import numpy as np
import torch
from grav_sim import settings
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import ManoLayer, MANOOutput
from trimesh import Trimesh as Mesh

from grav_sim.geom import aligning_rotation
from grav_sim.viz import mesh_plot

dip_indexes = [3, 6, 12, 9, 15]
digit_mcp_indexes = [1, 4, 10, 7]
digit_mcp_indexes = [1, 4, 10, 7]
tip_indexes = [17, 18, 19, 20, 16]
parents = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]
mano_joint_order = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
segment_names = [f'finger_{i}' for i in range(5)] + ['palm']
segment_joints = dict(
    finger_0=[13, 14, 15, 16],
    finger_1=[1, 2, 3, 17],
    finger_2=[4, 5, 6, 18],
    finger_3=[10, 11, 12, 19],
    finger_4=[7, 8, 9, 20])
finger_labels = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb']


def joint_to_finger_label(joint):
    for i, (finger_n, joints) in enumerate(segment_joints.items()):
        if joint in joints:
            return finger_labels[i]
    return 'wrist'


segment_faces = dict()
hand_segments_assets_path = pathlib.Path(settings.hand_segments_assets_path)
for segment_path in hand_segments_assets_path.glob('*_faces.npy'):
    _segment_name = '_'.join(segment_path.name.split('_')[:-1])
    segment_faces[_segment_name] = np.load(segment_path)

anatomy_loss_fn = AnatomyConstraintLossEE(reduction='sum')
anatomy_loss_fn.setup()


def _loss_fn(predicted, actual, joint_eueler_angles, reg_weight=.0001):
    mse_loss = torch.mean((predicted - actual) ** 2)
    anatomy_loss = anatomy_loss_fn(joint_eueler_angles) * reg_weight
    return mse_loss + anatomy_loss


def create_mano_layer():
    return ManoLayer(
        center_idx=None,
        mano_assets_root=settings.mano_assets_path,
        use_pca=False,
        rot_mode='axisang',
        side='right',
        flat_hand_mean=True)


def joint_to_limb_name(joint_index):
    joint_index = int(joint_index)
    for key, joints in segment_joints.items():
        if joint_index in joints:
            return key
    return None


def joints_until_root(joint):
    joint_path = list()
    parent = parents[joint]
    while parent:
        joint_path.append(parent)
        parent = parents[parent]
    return joint_path


def joints_until_leaf(joint):
    joint_path = list()
    while True:
        try:
            joint = parents.index(joint)
            joint_path.append(joint)
        except ValueError:
            return joint_path[:-1]


def joint_chain(joint):
    up = joints_until_root(joint)
    down = joints_until_leaf(joint)
    return list(reversed(down)) + [joint] + up


class Hand:

    def __init__(self, mano_layer: ManoLayer = None, closed_mesh=True):
        if mano_layer is None:
            mano_layer = create_mano_layer()
        self.mano_layer = mano_layer
        self.axis_fk = AxisLayerFK(mano_assets_root=mano_layer.mano_assets_root)

        self.shape_params = torch.zeros((1, 10), requires_grad=True).float()
        self.pose_params = torch.zeros((1, 48), requires_grad=True).float()
        self.anatomical_pose_params = torch.zeros((1, 48), requires_grad=False).float()

        self.vertices = None
        self.joints = None
        if closed_mesh:
            self.faces = mano_layer.get_mano_closed_faces().detach().numpy()
        else:
            self.faces = mano_layer.th_faces

        self.score(self.pose_params, self.shape_params)

    def center_at_wrist(self):
        self.vertices -= self.joints[0]
        self.joints -= self.joints[0]

    def score(self, pose_params=None, shape_params=None):

        if pose_params is not None:
            self.pose_params = pose_params.clone().detach() if torch.is_tensor(pose_params) else torch.Tensor(
                pose_params)
            self.pose_params.requires_grad_(True)

        if shape_params is not None:
            self.shape_params = shape_params.clone().detach() if torch.is_tensor(shape_params) else torch.Tensor(
                shape_params)
            self.shape_params.requires_grad_(True)

        mano_output = self.mano_layer.forward(self.pose_params, self.shape_params)
        self.anatomical_pose_params = self.axis_fk(mano_output.transforms_abs)[-1].detach()
        self.anatomical_pose_params[0, 0] = self.pose_params[0, :3]  # pass along wrist rotation

        self.vertices = mano_output.verts.detach().numpy()[0]
        self.joints = mano_output.joints.detach().numpy()[0]
        # Original MANO joint order
        # https://github.com/lixiny/manotorch/blob/933be97d6fa05c729656669640084251ce644b0a/manotorch/manolayer.py#L239
        self.joints = self.joints[mano_joint_order, :]

        '''
        if mano_output.center_idx is not None:
            center = self.joints[mano_output.center_idx]
            self.vertices -= center
            self.joints -= center
        '''
        return mano_output

    def compose(self, anatomical_pose_params):
        anatomical_pose_params = torch.Tensor(anatomical_pose_params).reshape((1, 16, 3))
        pose_params = self.axis_fk.compose(anatomical_pose_params).reshape((1, -1))
        # pass along wrist rotation
        pose_params[0, :3] = anatomical_pose_params[0, 0]
        wrist = self.joints[0]
        mano_output = self.score(pose_params, self.shape_params)
        translation = wrist - self.joints[0]
        self.joints += translation
        self.vertices += translation
        return mano_output

    def fit_mano(self, mano_output: MANOOutput):
        return self.score(
            mano_output.full_poses,
            mano_output.betas)

    def fit_joints(self, target_joints, budget=1000, max_loss=3e-05, loss_fn=_loss_fn):
        target_joints = torch.from_numpy(target_joints)
        optimizer = torch.optim.Adam([self.pose_params])
        for i in range(budget):
            optimizer.zero_grad()
            mano_out = self.mano_layer.forward(self.pose_params)
            joint_euler_angles = self.axis_fk(mano_out.transforms_abs)[-1]
            loss = loss_fn(mano_out.joints, target_joints, joint_euler_angles)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
            if loss.item() < max_loss:
                break

        return self.score(
            mano_out.full_poses,
            mano_out.betas)

    def core_mesh(self):
        _segment_faces = segment_faces['palm']
        mesh = self.mesh()
        return mesh.submesh(_segment_faces, only_watertight=False, append=True)

    def limb_mesh(self, segment):
        _segment_faces = segment_faces[segment]
        mesh = self.mesh()
        return mesh.submesh(_segment_faces, only_watertight=False, append=True)

    def limb_mesh_complement(self, segment):
        return self.limb_mesh(segment + '_remaining')

    def affine_align_with(self, other):
        other: Hand

        to_center = -self.joints[0]
        other_wrist = other.joints[0]
        self.center_at_wrist()
        other.center_at_wrist()

        align_dorsals = aligning_rotation(self.get_dorsal_normal(), other.get_dorsal_normal()).transpose()
        self.vertices = self.vertices @ align_dorsals
        self.joints = self.joints @ align_dorsals

        align_indexes = aligning_rotation(self.index_finger_direction(), other.index_finger_direction()).transpose()
        self.vertices = self.vertices @ align_indexes
        self.joints = self.joints @ align_indexes

        self.vertices += other_wrist
        self.joints += other_wrist

        A_center = np.eye(4)
        A_center[:-1, -1] = to_center

        B_align_dorsals = np.eye(4)
        B_align_dorsals[:-1, :-1] = align_dorsals

        C_align_indexes = np.eye(4)
        C_align_indexes[:-1, :-1] = align_indexes

        D_align_rotations = C_align_indexes.T @ B_align_dorsals.T

        E_to_wrist = np.eye(4)
        E_to_wrist[:-1, -1] = other_wrist

        return E_to_wrist @ D_align_rotations @ A_center

    def index_finger_direction(self):
        wrist_0 = self.joints[0]
        mcp_1 = self.joints[1]
        wrist_to_mcp_1 = mcp_1 - wrist_0
        return wrist_to_mcp_1 / np.linalg.norm(wrist_to_mcp_1)

    def little_finger_direction(self):
        wrist_0 = self.joints[0]
        mcp_4 = self.joints[4]
        wrist_to_mcp_4 = mcp_4 - wrist_0
        return wrist_to_mcp_4 / np.linalg.norm(wrist_to_mcp_4)

    def get_dorsal_normal(self, left_hand=False):
        index = self.index_finger_direction()
        little = self.little_finger_direction()
        normal = np.cross(little, index)
        if left_hand:
            normal = -normal
        return normal / np.linalg.norm(normal)

    def mesh(self):
        return Mesh(
            self.vertices,
            self.faces)

    def mesh_plot(self, color=None, opacity=.5, **kwargs):
        return mesh_plot(self.mesh(), color=color, opacity=opacity, **kwargs)


def standard_orientation(body, obj_mesh):
    mano_layer = create_mano_layer()
    standard_hand = Hand(mano_layer)
    standard_hand.center_at_wrist()
    transformation = body.affine_align_with(standard_hand)
    obj_mesh.apply_transform(transformation)
    return transformation, standard_hand
