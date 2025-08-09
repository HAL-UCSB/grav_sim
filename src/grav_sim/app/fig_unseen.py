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


selected_object = st.selectbox('Object', ycb_aff.list_object_names())
paths = ycb_aff.list_scene_paths(object_name=selected_object)
body = hands.Hand()
for path in paths:
    scene_mano, obj_mesh = ycb_aff.load_scene(path)
    body.fit_mano(scene_mano)
    fig = viz.figure(
        viz.mesh_plot(obj_mesh, color='white', opacity=1, lighting=dict(ambient=0.55)),
        body.mesh_plot(color='white', opacity=1, lighting=dict(ambient=0.55)),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ))
    st.plotly_chart(fig)