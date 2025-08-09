import datetime
import json

import numpy as np
import pandas as pd
from trimesh.collision import CollisionManager

from grav_sim import viz
from grav_sim.geom import erode, nearest_neighbor
from grav_sim.hands import Hand, joint_to_limb_name
from grav_sim.viz import scatter_plot, mesh_plot

xyz_columns = list('xyz')
pose_columns = [f'pose_{i}' for i in range(48)]
joint_column = 'joint'


def _generate_poses(joint, pose, step):
    poses = list()
    for i in range(3):
        up = pose.copy()
        down = pose.copy()
        up[0, joint, i] += step
        down[0, joint, i] -= step
        poses.append(up)
        poses.append(down)
    return poses


class Simulation:

    def __init__(self, body: Hand, obj_mesh, rom):
        self.body = body
        self.seed_pose = self.body.anatomical_pose_params.detach().numpy()

        self.pcd_size = 0
        self.pcd = None
        self.poses = None
        self.joints = None
        self.obj_mesh = obj_mesh
        self.rom = rom
        self.frames = list()
        self.frame_titles = list()

        hand_mesh = body.mesh()
        self.hand_mesh = hand_mesh
        self.obj_mesh, self.contact_surface = erode(obj_mesh, hand_mesh)
        self.object_collisions = CollisionManager()
        self.object_collisions.add_object('object', self.obj_mesh)

        self._params = dict()

        assert not self.object_collisions.in_collision_single(self.body.mesh())

    def _is_outside_rom(self, joint, pose, rom_tolerance=np.deg2rad(5)):
        angle_index = joint * 3
        angles_slice = slice(angle_index, angle_index + 3)
        floor, ceil = self.rom[:, angles_slice]
        rot = pose[0, joint]
        beyond_floor = rot < (floor - rom_tolerance)
        beyond_ceil = rot > (ceil + rom_tolerance)
        return np.any(beyond_floor | beyond_ceil)

    def _record_animation_frame(self):
        _pcd = self.pcd[:self.pcd_size, :]
        self.frames.append([
            mesh_plot(self.obj_mesh),
            self.body.mesh_plot(),
            scatter_plot(_pcd, colors=np.array([1, 0, 0] * len(_pcd)))
        ])

    def simulate(self, max_pcd_size, step_size, kinematic_chain, tracked_joint_index=-1, sparsity=None, randomize=True,
                 rom_tolerance=np.deg2rad(5), record_animation=False, verbose=False, progress=None):

        # update simulation params to save as metadata later
        now = datetime.datetime.now(datetime.UTC)
        self._params = dict(
            utc_timestamp=now.timestamp(),
            tracked_joint=kinematic_chain[tracked_joint_index],
            args=[max_pcd_size, step_size, kinematic_chain],
            kwargs=dict(
                tracked_joint_index=tracked_joint_index,
                sparsity=sparsity,
                randomize=randomize,
                rom_tolerance=rom_tolerance)
        )

        # initialize buffers
        self.pcd_size = 0
        self.pcd = np.empty((int(max_pcd_size), 3), dtype=float)
        self.poses = np.empty((int(max_pcd_size), 48), dtype=float)
        self.joints = np.empty((int(max_pcd_size), 1), dtype=int)
        self.frames = []
        visits = set()

        # initialize collision manager
        tracked_joint = kinematic_chain[tracked_joint_index]
        kinematic_chain = sorted(kinematic_chain)
        leaf = kinematic_chain[-2]
        segment = joint_to_limb_name(leaf)
        segment_mesh = self.body.limb_mesh(segment)
        core_mesh = self.body.core_mesh()
        core_mesh = erode(core_mesh, segment_mesh)[0]
        limb_collisions = CollisionManager()
        limb_collisions.add_object('core', core_mesh)

        jobs = [(leaf, self.seed_pose)]
        while jobs:
            pop_index = np.random.randint(len(jobs)) if randomize else -1

            joint, pose = jobs.pop(pop_index)
            if verbose and self.pcd_size % 10 == 0:
                print(f'pcd size: {self.pcd_size}\tjobs: {len(jobs)}')
            if progress:
                progress(float(self.pcd_size / max_pcd_size))

            # finishes when the point cloud is large enough
            if max_pcd_size is not None and self.pcd_size >= max_pcd_size:
                jobs.clear()
                break

            # skip visited poses
            pose_code = pose.tobytes()
            if pose_code in visits:
                continue
            visits.add(pose_code)

            # check pose validity if this is not the initial pose
            if self.pcd_size > 0:

                # skip poses outside the ROM of the current joint
                if self._is_outside_rom(joint, pose, rom_tolerance):
                    continue

                # set the body pose
                self.body.compose(pose)

                tracked_position = self.body.joints[tracked_joint]
                _pcd = self.pcd[:self.pcd_size, :]
                closest = nearest_neighbor(tracked_position, _pcd)[-1]
                if closest < sparsity:
                    continue

                # skip if the hand collides with the object
                hand_mesh = self.body.mesh()
                if self.object_collisions.in_collision_single(hand_mesh):
                    continue

                # skip self-colliding poses
                segment_mesh = self.body.limb_mesh(segment)
                if limb_collisions.in_collision_single(segment_mesh):
                    continue

            # add to the pcd
            tracked_position = self.body.joints[tracked_joint].copy()
            self.pcd[self.pcd_size] = tracked_position
            self.poses[self.pcd_size] = pose.flatten().copy()
            self.joints[self.pcd_size] = tracked_joint
            self.pcd_size += 1

            # record animation frame
            if record_animation:
                self._record_animation_frame()

            # push next poses
            for link in kinematic_chain[:-1]:
                next_poses = _generate_poses(link, pose, step_size)
                jobs += [(link, next_pose) for next_pose in next_poses]

            # auto set sparsity: conservative estimate of the minimum distance traversed by a step
            if sparsity is None:
                distances = []
                for close_pose in _generate_poses(joint, pose, step_size):
                    self.body.compose(close_pose)
                    distance = np.linalg.norm(self.body.joints[tracked_joint] - tracked_position)
                    distances.append(distance)
                # safety margin
                sparsity = np.median(distances) * .975

        self.pcd = self.pcd[:self.pcd_size, :].copy()
        self.poses = self.poses[:self.pcd_size, :].copy()
        self.joints = self.joints[:self.pcd_size, :].copy()
        self.body.compose(self.seed_pose)
        if progress:
            progress(1.0)

    def get_result_df(self):
        result = pd.DataFrame(self.pcd, columns=xyz_columns)
        result[pose_columns] = self.poses
        result[joint_column] = self.joints
        return result

    def save_results(self, dir_path, filename=None):
        result = self.get_result_df()
        result_path = dir_path / (filename or 'simulation.csv')
        result.to_csv(result_path, index_label='sim_index')
        params_path = dir_path / 'params.json'
        if not params_path.exists():
            with params_path.open('w') as file:
                json.dump(self._params, file)
