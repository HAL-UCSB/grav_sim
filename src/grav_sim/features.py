import pathlib

import numpy as np
import pandas as pd
from trimesh import Trimesh

from grav_sim import costs, ycb_aff, hands, viz, geom


def distance_from_body_surface(body_mesh=None, target=None, **kwargs):
    return costs.distance_from_surface(body_mesh, target)


def shortest_distance_from_body_surface(body_mesh=None, target=None, **kwargs):
    return distance_from_body_surface(body_mesh=body_mesh, target=target, **kwargs)[0]


def smallest_distance_from_object_surface(obj_mesh=None, target=None, **kwargs):
    return costs.distance_from_surface(obj_mesh, target)[0]


def distance_from_reach_convex_hull(target=None, finger_pcd=None, **kwargs):
    return costs.distance_from_convex_hull(target, finger_pcd).sum()


def distance_from_reach_alpha_shape(target=None, pcd=None, **kwargs):
    return costs.distance_from_alpha_shape(target, pcd)[0]


def signed_distance_from_reach_convex_hull(target=None, pcd=None, **kwargs):
    return costs.signed_distance_from_convex_hull(target, pcd)


def signed_distance_from_reach_alpha_shape(target=None, pcd=None, **kwargs):
    return costs.signed_distance_from_alpha_shape(target, pcd)


def anatomy_loss(target_pose=None, **kwargs):
    return costs.anatomy_loss(target_pose).item()


def x_displacement_from_fingertip_to_target(initial_joint_position=None, target=None, **kwargs):
    return (initial_joint_position - target).flatten()[0]


def y_displacement_from_fingertip_to_target(initial_joint_position=None, target=None, **kwargs):
    return (initial_joint_position - target).flatten()[1]


def z_displacement_from_fingertip_to_target(initial_joint_position=None, target=None, **kwargs):
    return (initial_joint_position - target).flatten()[2]


def distance_from_initial_fingertip_to_target(initial_joint_position=None, target=None, **kwargs):
    return costs.euclidean_distance(
        initial_joint_position, target)[0]


def sum_of_quaternion_distances_from_initial_pose_to_target_pose(initial_pose=None, target_pose=None, **kwargs):
    return costs.quaternion_distance(initial_pose, target_pose).sum()


def sum_of_absolute_differences_from_initial_pose_to_target_pose(initial_pose=None, target_pose=None, **kwargs):
    return costs.sum_of_absolute_differences(initial_pose, target_pose, axis=None)


def sum_of_euclidean_distances_of_euler_angles_from_initial_pose_to_target_pose(initial_pose=None, target_pose=None,
                                                                                **kwargs):
    return costs.euclidean_distance(initial_pose, target_pose).sum()


def volume_of_finger_reachable_space(finger_pcd=None, **kwargs):
    return costs.convex_hull_volume(finger_pcd).item()


def volume_of_grasp_reachable_space(grasp_pcd=None, **kwargs):
    return costs.convex_hull_volume(grasp_pcd).item()


def rechable_volume_ratio(finger_pcd=None, grasp_pcd=None, **kwargs):
    return (costs.convex_hull_volume(finger_pcd) / costs.convex_hull_volume(grasp_pcd)).item()


def cage_ratio(obj_mesh=None, body_mesh=None, **kwargs):
    return costs.cage_ratio(obj_mesh, body_mesh)


def target_x(target=None, **kwargs):
    return target.flatten()[0]


def target_y(target=None, **kwargs):
    return target.flatten()[1]


def target_z(target=None, **kwargs):
    return target.flatten()[2]


def initial_fingertip_x(initial_joint_position=None, **kwargs):
    return initial_joint_position.flatten()[0]


def initial_fingertip_y(initial_joint_position=None, **kwargs):
    return initial_joint_position.flatten()[1]


def initial_fingertip_z(initial_joint_position=None, **kwargs):
    return initial_joint_position.flatten()[2]


def reach_volume(pcd=None, **kwargs):
    return costs.convex_hull_volume(pcd)


def free_hand_finger_individuation_index(joint=None, **kwargs):
    # Table 1
    # https://doi.org/10.1523/JNEUROSCI.20-22-08542.2000
    fii = {
        16: 0.983,
        17: 0.982,
        18: 0.937,
        19: 0.907,
        20: 0.943
    }
    return fii[joint]


def grasping_hand_individuation_index(joint=None, **kwargs):
    # Section 5.2.1
    # https://doi.org/10.1145/3411764.3445197
    fii = {
        16: .98,
        17: .97,
        18: .96,
        19: .95,
        20: .96
    }
    return fii[joint]


def solofinger_ease_of_use(joint=None, **kwargs):
    # Section 5.2.2
    # https://doi.org/10.1145/3411764.3445197
    return {
        16: 4,
        17: 5,
        18: 3,
        19: 1,
        20: 2
    }[joint]


features = [
    target_x,
    target_y,
    target_z,
    initial_fingertip_x,
    initial_fingertip_y,
    initial_fingertip_z,
    x_displacement_from_fingertip_to_target,
    y_displacement_from_fingertip_to_target,
    z_displacement_from_fingertip_to_target,
    distance_from_initial_fingertip_to_target,
    shortest_distance_from_body_surface,
    smallest_distance_from_object_surface,
    sum_of_quaternion_distances_from_initial_pose_to_target_pose,
    sum_of_euclidean_distances_of_euler_angles_from_initial_pose_to_target_pose,
    # sum_of_absolute_differences_from_initial_pose_to_target_pose,

    volume_of_finger_reachable_space,
    rechable_volume_ratio,
    cage_ratio,

    distance_from_reach_convex_hull,

    anatomy_loss,
    free_hand_finger_individuation_index,
    grasping_hand_individuation_index,
    solofinger_ease_of_use
]

feat_names = [feat.__name__ for feat in features]


def apply_feature(feature, body: hands.Hand, obj_mesh: Trimesh, poses, reach_pcd):
    pass


def graspr_features(body, obj_mesh, joint, target_pose, target, finger_pcd, grasp_pcd, make_standard_pose=False):
    if make_standard_pose:
        transformation, _ = hands.standard_pose(body, obj_mesh)
        pcd_hom = geom.cartesian_to_homogeneous(finger_pcd)
        pcd_hom = pcd_hom @ transformation.T
        finger_pcd = geom.homogeneous_to_cartesian(pcd_hom)

        pcd_hom = geom.cartesian_to_homogeneous(grasp_pcd)
        pcd_hom = pcd_hom @ transformation.T
        grasp_pcd = geom.homogeneous_to_cartesian(pcd_hom)

    joint = int(joint)
    limb_name = hands.joint_to_limb_name(joint)
    complement_body_mesh = body.limb_mesh_complement(limb_name)

    feat_kwargs = dict(
        obj_mesh=obj_mesh,
        initial_pose=body.pose_params.detach().numpy(),
        body_mesh=complement_body_mesh,
        initial_joint_position=body.joints[joint].reshape(-1, 3).astype(float),
        grasp_pcd=grasp_pcd,
        finger_pcd=finger_pcd,
        target=target.reshape(-1, 3).astype(float),
        target_pose=target_pose.reshape(-1, 16, 3).astype(float),
        joint=joint
    )

    feat_vals = []
    for feat_func in features:
        feat_val = feat_func(**feat_kwargs)
        feat_vals.append(feat_val)

    return np.array(feat_vals)


if __name__ == '__main__':

    xyz = ['x', 'y', 'z']
    a_pose_columns = [f'a_pose_{i}' for i in range(48)]
    b_pose_columns = [f'b_pose_{i}' for i in range(48)]
    a_target_columns = [f'a_{i}' for i in xyz]
    b_target_columns = [f'b_{i}' for i in xyz]

    df_0 = pd.read_csv('batch_0.csv')
    df_1 = pd.read_csv('batch_1.csv')

    df_0.columns = [column.lower() for column in df_0.columns]
    df_0.rename(
        inplace=True,
        columns={
            'a_index': 'a_sim_index',
            'b_index': 'b_sim_index'
        })
    # df = pd.concat([df_0, df_1]).copy()
    df = pd.read_csv('clean_dataset.csv')

    body = hands.Hand(hands.create_mano_layer(mano_assets=_settings.mano_assets))
    mano_layer = ycb_aff.create_mano_layer(mano_assets=_settings.mano_assets)
    cached_scene = None
    df = df.sort_values(by=['a_scene', 'b_scene']).reset_index().copy()
    df[[feat.__name__ for feat in features]] = 0.0

    simulations_path = pathlib.Path(_settings.assets / 'simulations')
    switching_scenes = False

    for i, row in df.iterrows():
        assert (df.iloc[i] == row).all()

        # Load scene
        if cached_scene is None or cached_scene != row.a_scene:
            print('switching scenes')
            switching_scenes = True
            cached_scene = row.a_scene
            scene_path = _settings.ycb_aff_assets / 'grasps' / f'{row.a_scene}.pickle'
            scene_mano, obj_mesh = ycb_aff.load_scene(
                scene_path,
                mano_layer,
                ycb_aff_assets_path=_settings.ycb_aff_assets)
            body.fit_mano(scene_mano)
            transformation, _ = standard_pose(body, obj_mesh)
            feat_kwargs = dict(
                obj_mesh=obj_mesh,
                initial_pose=body.pose_params.detach().numpy())

        debug_pcds = []
        delta = np.zeros(len(features))
        for prefix in ['a', 'b']:
            scene = row[f'{prefix}_scene']
            joint = row[f'{prefix}_joint']
            target_columns = a_target_columns if prefix == 'a' else b_target_columns
            poses_columns = a_pose_columns if prefix == 'a' else b_pose_columns

            target = row[target_columns].to_numpy().astype(float)
            target_pose = row[poses_columns].to_numpy().astype(float)

            pcd_path = simulations_path / scene / str(joint) / f'{scene}_{joint}.csv'
            pcd = pd.read_csv(pcd_path)[xyz]
            pcd_hom = geom.cartesian_to_homogeneous(pcd)
            pcd_hom = pcd_hom @ transformation.T
            pcd = geom.homogeneous_to_cartesian(pcd_hom)
            debug_pcds.append(pcd.copy())

            feat_values = graspr_features(body, obj_mesh, joint, target_pose, target, pcd)
            delta = np.array(feat_values) - delta

        df.loc[i, feat_names] = delta
        print(i, f'{100 * i / len(df):.2f}%')

        if switching_scenes:
            switching_scenes = False
            viz.figure(
                viz.mesh_plot(obj_mesh),
                body.mesh_plot(),
                viz.scatter_plot(debug_pcds[0]),
                viz.scatter_plot(debug_pcds[1])
            ).show()

    df.to_csv('bigtable3.csv')
    print('ok')
