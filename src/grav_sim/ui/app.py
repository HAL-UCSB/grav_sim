import os
import pathlib

import PIL.Image
import torch

torch.classes.__path__ = [pathlib.Path(torch.__path__[0]) / torch.classes.__file__]

from six import BytesIO

os.environ.update(dict(
    mano_assets='assets/mano',
    ycb_aff_assets='assets/ycb_aff',
    rom_csv='assets/eatonhand_rom.csv',
    hand_segments_assets='assets/hand_segments'))

import torch
import pathlib

torch.classes.__path__ = [pathlib.Path(torch.__path__[0]) / torch.classes.__file__]

import numpy as np
import pandas as pd
import streamlit as st

from grav_sim import ycb_aff
from grav_sim import viz
from grav_sim import hands
from grav_sim import settings
from grav_sim import sim


@st.cache_resource
def get_rom():
    df_rom = pd.read_csv(settings.rom_csv)
    _rom = np.stack((df_rom.floor.values, df_rom.ceil.values))
    return np.deg2rad(_rom)


body = hands.Hand()

if 'pcd_full' not in st.session_state:
    st.session_state.pcd_full = []
    st.session_state.frames = []
    st.session_state.poses = []

horizontal = iter(st.columns(4))

with next(horizontal):
    degrees = st.number_input('ROM tolerance (degrees)',
                              value=10.0,
                              min_value=0.0,
                              max_value=360.0,
                              step=1.0,
                              key='number_input_rom',
                              help='Enables joint movement beyond the range-of-motion defined in the assets.'
                                   'Useful if the scene’s initial hand pose exceeds the default joint limits.')
    rom_tolerance = np.deg2rad(degrees)

with next(horizontal):
    degrees = st.number_input('Step angle (degrees)',
                              value=10.0,
                              min_value=0.0,
                              max_value=360.0,
                              step=1.0,
                              key='number_input_step',
                              help='Controls how much joint angles change per simulation step.'
                                   'Lower values yield denser point clouds and more accurate collisions, but slower performance.'
                                   'Higher values run faster but may reduce collision accuracy.')
    step_size = np.deg2rad(degrees)

with next(horizontal):
    max_pcd_size = st.number_input('Max point cloud size',
                                   value=int(1e3),
                                   min_value=1,
                                   step=1,
                                   key='number_input_pcd_size',
                                   help='Sets the maximum size of the generated point cloud.'
                                        'The actual size may be smaller if few valid hand poses are found.'
                                        'Larger values require longer computation time.')

with next(horizontal):
    randomize = st.checkbox('Randomize',
                            value=True,
                            help='Randomizes the order of hand pose exploration during simulation.'
                                 'Helps promote broader coverage and prevents clustering of points in limited regions of the space.')

selected_finger_labels = st.multiselect('Tracked fingertip', hands.finger_labels, hands.finger_labels)
selected_fingertips = [hands.tip_indexes[hands.finger_labels.index(label)] for label in selected_finger_labels]

scenes = ycb_aff.list_scene_paths()
if scene := st.selectbox('scene', scenes):
    scene_mano, obj_mesh = ycb_aff.load_scene(scene)
    body.fit_mano(scene_mano)
    st.plotly_chart(
        viz.figure(
            viz.mesh_plot(obj_mesh),
            body.mesh_plot()))

generate_gif = st.checkbox('Generate GIF', True)
if st.button('Run simulation'):
    st.session_state.pcd_full.clear()
    st.session_state.frames.clear()
    st.session_state.poses.clear()

    rom = get_rom()
    simulation = sim.Simulation(
        body,
        obj_mesh,
        rom)

    with st.spinner("Simulating...", show_time=True):
        for fingertip in selected_fingertips:
            simulation.simulate(
                max_pcd_size,
                step_size,
                rom_tolerance=rom_tolerance,
                kinematic_chain=hands.joints_until_root(fingertip) + [fingertip],
                randomize=randomize,
                record_animation=generate_gif,
                verbose=True)
            st.session_state.pcd_full += simulation.pcd.tolist()
            st.session_state.poses += simulation.poses.tolist()
            st.session_state.frames += simulation.frames

    if st.session_state.frames:
        progress_bar = st.progress(0, 'Generating GIF')
        all_frames = []
        for i, frame in enumerate(st.session_state.frames):
            fig = viz.figure(*frame)
            img = PIL.Image.open(BytesIO(fig.to_image()))
            all_frames.append(img)
            progress_value = float(i) / len(st.session_state.frames)
            progress_bar.progress(progress_value, f'Generating GIF ({i} of {len(st.session_state.frames)} frames)')

        all_frames[0].save(
            'simulation.gif',
            save_all=True,
            append_images=all_frames[1:],
            duration=1_000.0 / 5,
            loop=0)

        progress_bar.progress(1)
        progress_bar.empty()

if st.session_state.pcd_full:
    fig = viz.figure(
        viz.mesh_plot(obj_mesh),
        body.mesh_plot(),
        viz.scatter_plot(st.session_state.pcd_full)
    )

    st.download_button(
        label='Download point cloud as CSV',
        data=pd.DataFrame(
            st.session_state.filtered_pcd if 'filtered_pcd' in st.session_state else st.session_state.pcd_full).to_csv(
            index=False).encode('utf-8'),
        file_name=f'{scene}.csv',
        mime='text/csv')

    if st.session_state.frames:
        st.image('simulation.gif')

    st.plotly_chart(fig)
