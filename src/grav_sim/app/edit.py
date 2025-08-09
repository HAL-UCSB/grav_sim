import math
import pathlib

import numpy as np
import streamlit as st
import pandas as pd
from alphashape import alphashape
from grav_sim import hands, viz, ycb_aff, costs, settings, sim
from grav_sim import features_2 as features
from grav_sim.viz import scatter_plot
import plotly.express as px

results_path = pathlib.Path(
    st.text_input('Simulation result CSV') or r'C:\Users\Arthur\Downloads\obj_0_grasp_0.csv')
scene_path = settings.ycb_aff_assets / 'grasps' / f'{results_path.stem}.pickle'


def validate_file(path, extension='csv'):
    if not path.exists() or not path.name.endswith(f'.{extension}'):
        st.error(f'"{path}" must be an existing {extension.capitalize()} file')
        return False
    return True


feature_map = {feat_fn.__name__.replace('_', ' '): feat_fn for feat_fn in features.features}

selected_features = st.multiselect('Features', feature_map.keys())

if st.button('Calculate Features') and validate_file(results_path):
    results = pd.read_csv(results_path)
    pcd = results[sim.xyz_columns].to_numpy()
    poses = results[sim.pose_columns].to_numpy()
    joints = results[sim.joint_column].to_numpy()

    scene_mano, obj_mesh = ycb_aff.load_scene(scene_path)
    body = hands.Hand()
    body.compose(poses[0])

    for feat_name in selected_features:
        feat_fn = feature_map[feat_name]
        feat_values = feat_fn(results, body, obj_mesh)
        results[feat_name] = feat_values

        st.plotly_chart(
            viz.figure(
                viz.mesh_plot(obj_mesh),
                body.mesh_plot(),
                scatter_plot(pcd, colors=feat_values),
                title=feat_name
            )
        )

if st.button('new fig'):

    results = pd.read_csv(results_path)
    pcd = results[sim.xyz_columns].to_numpy()
    poses = results[sim.pose_columns].to_numpy()
    joints = results[sim.joint_column].to_numpy()

    scene_mano, obj_mesh = ycb_aff.load_scene(scene_path)
    body = hands.Hand()
    body.compose(poses[0])
    feat_name = 'grasping_microgesture_region'.replace('_', ' ')

    feat_fn = feature_map[feat_name]
    feat_values = feat_fn(results, body, obj_mesh)
    results[feat_name] = feat_values

    plots = [
        viz.mesh_plot(obj_mesh),
        body.mesh_plot()
    ]
    names = ['On Object', 'On Body', 'In Air Mid', 'In Air Far']
    for i in range(4):
        mask = feat_values == i
        colors = np.array([px.colors.qualitative.Plotly[i]] * mask.sum())
        plot = viz.scatter_plot(pcd[mask], colors=colors, opacity=1, name=names[i])
        plots.append(plot)

    st.plotly_chart(
        viz.figure(*plots),
        title=feat_name)

    # for i in range(len(pcd)):
    #     pose_idx = np.where(not np.isclose((rest_pose - poses[i]), 0))
    #     moving_joint = math.floor(pose_idx / 3)
    #     fingertip = hands.joints_until_leaf(moving_joint)[-1]
    #     limb_name = hands.joint_to_limb_name(moving_joint)
    #     complement_body_mesh = body.limb_mesh_complement(limb_name)
    #
    #     feat_kwargs = dict(
    #         obj_mesh=obj_mesh,
    #         initial_pose=body.pose_params.detach().numpy(),
    #         body_mesh=complement_body_mesh,
    #         #initial_joint_position=body.joints[joint].reshape(-1, 3).astype(float),
    #         grasp_pcd=pcd,
    #         #finger_pcd=finger_pcd,
    #         #target=target.reshape(-1, 3).astype(float),
    #         target_pose=target_pose.reshape(-1, 16, 3).astype(float),
    #         joint=joint
    #     )
