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
    st.text_input('Simulation result CSV') or r'C:\Users\Arthur\Downloads\obj_34_grasp_-4.csv')
scene_path = settings.ycb_aff_assets / 'grasps' / f'{results_path.stem}.pickle'


def validate_file(path, extension='csv'):
    if not path.exists() or not path.name.endswith(f'.{extension}'):
        st.error(f'"{path}" must be an existing {extension.capitalize()} file')
        return False
    return True


feature_map = {feat_fn.__name__.replace('_', ' '): feat_fn for feat_fn in features.features}

selected_finger_labels = st.multiselect('Tracked fingertip', hands.finger_labels, hands.finger_labels)
selected_fingertips = [hands.tip_indexes[hands.finger_labels.index(label)] for label in selected_finger_labels]

results = pd.read_csv(results_path)
results = results[np.isin(results[sim.joint_column], selected_fingertips)]
pcd = results[sim.xyz_columns].to_numpy()
poses = results[sim.pose_columns].to_numpy()
joints = results[sim.joint_column].to_numpy()

scene_mano, obj_mesh = ycb_aff.load_scene(scene_path)
body = hands.Hand()
body.compose(poses[0])

features = [
    features.on_object,
    features.on_body,
    features.on_air_mid,
    features.on_air_far
]

plots = [
    viz.mesh_plot(obj_mesh, color='white', opacity=1, lighting=dict(ambient=0.55)),
    body.mesh_plot(color='white', opacity=1, lighting=dict(ambient=0.55)),
]

for i, feat_fn in enumerate(features):
    feat_values = feat_fn(results, body, obj_mesh)
    break
    plot = viz.scatter_plot(pcd, colors=feat_values, opacity=.075, name=feat_fn.__name__)
    plots.append(plot)
    # color = px.colors.sample_colorscale('Turbo', .25 * i)[0]
    # rgb = color[len('rgb('):-2]
    # feat_values = feat_fn(results, body, obj_mesh) * 2
    # colors = np.array([f'rgba({rgb}, {v})'for v in feat_values])
    # plot = viz.scatter_plot(pcd, colors=colors, name=feat_fn.__name__)
    # plots.append(plot)

idx = st.selectbox('gesture', np.argsort(feat_values)[:10])
pose = poses[idx]
body.compose(pose)



fig = viz.figure(*plots)
fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=-1.5)))
fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
    ))

st.plotly_chart(
    fig,
    title=feat_fn.__name__)

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
