import pathlib
import pickle
import torch
import trimesh

import numpy as np
import pandas as pd

from types import SimpleNamespace
from manotorch.manolayer import ManoLayer
from grav_sim import settings

feix_taxonomy = [
    'Large Diameter',
    'Small Diameter',
    'Medium Wrap',
    'Adducted Thumb',
    'Light Tool',
    'Prismatic 4 Finger',
    'Prismatic 3 Finger',
    'Prismatic 2 Finger',
    'Palmar Pinch',
    'Power Disk',
    'Power Sphere',
    'Precision Disk',
    'Precision Sphere',
    'Tripod',
    'Fixed Hook',
    'Lateral',
    'Index Finger Extension',
    'Extension Type',
    'Distal Type',
    'Writing Tripod',
    'Tripod Variation',
    'Parallel Extension',
    'Adduction Grip',
    'Tip Pinch',
    'Lateral Tripod',
    'Sphere 4-Finger',
    'Quadpod',
    'Sphere Finger',
    'Stick',
    'Palmar',
    'Ring',
    'Ventral',
    'Inferior Pincer']


def create_object_grasp_df(data_path=settings.ycb_aff_assets_path, export_path=None):
    paths, objs, grasps = [], [], []

    for scene_path in list_scene_paths(data_path=data_path):
        with scene_path.open('rb') as file:
            scene_data = pickle.load(file, encoding='latin')
            ycb_obj = pathlib.Path(scene_data['body']).parents[1].name
            feix_grasp = scene_data['taxonomy']
            paths.append(scene_path)
            objs.append(ycb_obj)
            grasps.append(feix_grasp)

    df = pd.DataFrame(dict(
        path=paths,
        object=objs,
        grasp=grasps))

    df.sort_values('grasp', inplace=True)

    if export_path:
        df.to_csv(export_path, index=False)

    return df


def list_scene_paths(object_name=None, grasp_index=None, object_grasp_df=None, return_df=False,
                     data_path=settings.ycb_aff_assets_path):
    if object_name is None and grasp_index is None:
        grasps = pathlib.Path(data_path) / 'grasps'
        pattern = f'obj_*_grasp_*.pickle'
        gen = grasps.glob(pattern)
        return list(gen)

    if object_grasp_df is None:
        object_grasp_df = create_object_grasp_df(data_path=data_path)
    if object_name is None:
        mask = np.isin(object_grasp_df.grasp, grasp_index)
    else:
        mask = object_grasp_df.object.apply(lambda name: object_name in name)
        if grasp_index is not None:
            mask &= np.isin(object_grasp_df.grasp, grasp_index)
    if return_df:
        return object_grasp_df[mask]
    return object_grasp_df[mask].path.values.tolist()


def list_object_names(object_grasp_df=None, data_path=settings.ycb_aff_assets_path):
    if object_grasp_df is None:
        object_grasp_df = create_object_grasp_df(data_path=data_path)
    return list(sorted(set([name[4:] for name in np.unique(object_grasp_df.object)])))


def get_obj_mesh_path(scene_data):
    body = scene_data['body']
    path = pathlib.Path(body)
    path_parts = path.parts[-3:-1]
    path_parts = map(pathlib.Path, path_parts)
    middle = pathlib.Path.joinpath(*path_parts)
    return settings.ycb_aff_assets_path / 'models' / middle / 'textured.obj'


def create_mano_layer():
    return ManoLayer(
        center_idx=0,
        mano_assets_root=settings.mano_assets_path,
        side='right',
        use_pca=True,
        flat_hand_mean=True,
        ncomps=45)


def load_scene(pickle_path):
    """
    based on https://github.com/enriccorona/YCB_Affordance/blob/93b0d763267d9b105c4679f8eab6cff0c74b2ffa/visualize_grasps.py#L43
    """
    pickle_path = pathlib.Path(pickle_path)

    with pickle_path.open('rb') as file:
        scene_data = pickle.load(file, encoding='latin')

    pca_manorot = scene_data['pca_manorot']
    pca_poses = scene_data['pca_poses']
    pose_params = np.concatenate(([pca_manorot], pca_poses), 1)
    mano_trans = scene_data['mano_trans']

    mano_layer = create_mano_layer()
    mano_output = mano_layer.forward(
        torch.FloatTensor(pose_params),
        th_trans=torch.FloatTensor(mano_trans))

    mano_dict = mano_output._asdict()
    mano_dict['shape_params'] = np.zeros((1, 10))
    mano_dict['pose_params'] = pose_params
    mano_dict['vertices'] = mano_output.verts.cpu().data.numpy()[0]
    mano_dict['joints'] = mano_output.joints.cpu().data.numpy()[0]
    mano_dict['faces'] = mano_layer.th_faces.cpu().data.numpy()
    mano_dict['mano_layer'] = mano_layer
    custom_mano_output = SimpleNamespace(**mano_dict)

    obj_path = get_obj_mesh_path(scene_data)
    obj_mesh = trimesh.load(obj_path)
    obj_mesh.vertices -= np.array(mano_trans)

    return custom_mano_output, obj_mesh
