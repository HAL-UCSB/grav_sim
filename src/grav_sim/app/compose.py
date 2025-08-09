import math

import streamlit as st
import numpy as np
from grav_sim import hands, viz

st.title('MANOTorch AxisLayerFK Compose')

column = iter(st.columns(4))
display_origin = next(column).checkbox('Display origin', False)
display_wrist = next(column).checkbox('Display wrist', False)
hand_color = next(column).color_picker('Hand color')
opacity = next(column).slider('Hand opacity', min_value=0.0, max_value=1.0)


left, right = st.columns(2)
anatomical_pose_params = np.zeros((1, 16, 3))
segment_labels = ['Wrist', 'Index', 'Middle', 'Pinky', 'Ring', 'Thumb']
joint_labels = ['MCP', 'PIP', 'DIP']
xyz = 'xyz'
segment_containers = []
with left:
    for label in segment_labels:
        container = st.expander(label)
        segment_containers.append(container)

for i in range(16):
    segment_idx = int(math.ceil(float(i) / 3))
    joint_idx = (i - 1) % 3
    joint_label = joint_labels[joint_idx]
    segment_container = segment_containers[segment_idx]
    container = segment_container if i == 0 else segment_container.expander(joint_label)
    with container:
        for column, j in zip(container.columns(3), range(3)):
            joint_degrees = column.number_input(
                f'{xyz[j]}',
                value=0,
                min_value=-360,
                max_value=360,
                key=f'number_input_{segment_idx}_{joint_idx}_{j}')
            anatomical_pose_params[0, i, j] = np.deg2rad(joint_degrees)

with right:
    body = hands.Hand()
    body.compose(anatomical_pose_params)

    plots = [
        body.mesh_plot(color=hand_color, opacity=opacity)
    ]
    if display_origin:
        origin = viz.scatter_plot(np.zeros(3), colors=np.array([1, 0, 0]))
        plots.append(origin)
    if display_wrist:
        origin = viz.scatter_plot(body.joints[0], colors=np.array([1, 0, 0]))
        plots.append(origin)

    st.plotly_chart(viz.figure(*plots))
