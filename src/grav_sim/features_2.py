import numpy as np
import pandas as pd
from alphashape import alphashape
from trimesh import Trimesh

from grav_sim import hands, sim, costs
import plotly.express as px


def distance_from_initial_joint_position(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    joints = sim_results[sim.joint_column]

    initial_joint_position = {joint: body.joints[joint] for joint in np.unique(joints)}

    def calc(row):
        joint = row[sim.joint_column]
        current_position = row[sim.xyz_columns]
        origin = initial_joint_position[joint]
        return np.linalg.norm(origin - current_position)

    return sim_results.apply(calc, axis=1).values


def distance_from_joint_pcd_centroid(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    joints = sim_results[sim.joint_column]
    joint_pcd_centroids = dict()
    for joint in np.unique(joints):
        mask = sim_results[sim.joint_column] == joint
        centroid = sim_results[mask][sim.xyz_columns].mean()
        joint_pcd_centroids[joint] = centroid

    def calc(row):
        joint = row[sim.joint_column]
        current_position = row[sim.xyz_columns]
        centroid = joint_pcd_centroids[joint]
        return np.linalg.norm(centroid - current_position)

    return sim_results.apply(calc, axis=1).values


def distance_from_joint_pcd_medoid(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    joints = sim_results[sim.joint_column]
    joint_pcd_medoid = dict()
    for joint in np.unique(joints):
        mask = sim_results[sim.joint_column] == joint
        pcd = sim_results[mask][sim.xyz_columns]
        axis = 1 if pcd.shape[0] == 3 else 0
        medoid = np.median(pcd, axis=axis)
        joint_pcd_medoid[joint] = medoid

    def calc(row):
        joint = row[sim.joint_column]
        current_position = row[sim.xyz_columns]
        medoid = joint_pcd_medoid[joint]
        return np.linalg.norm(medoid - current_position)

    return sim_results.apply(calc, axis=1).values


def distance_from_body(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    pcd = sim_results[sim.xyz_columns].to_numpy()
    return costs.distance_from_surface(body.mesh(), pcd)


def distance_from_object(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    pcd = sim_results[sim.xyz_columns].to_numpy()
    return costs.distance_from_surface(obj_mesh, pcd)


def distance_from_pcd_hull(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh, alpha=1):
    pcd = sim_results[sim.xyz_columns].to_numpy()
    hull = alphashape(pcd, alpha)
    return costs.distance_from_surface(hull, pcd)


def on_object(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    on_obj = distance_from_object(sim_results, body, obj_mesh)
    on_obj = (on_obj - np.min(on_obj)) / (np.max(on_obj) - np.min(on_obj))
    return on_obj


def on_body(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    on_body = distance_from_body(sim_results, body, obj_mesh)
    on_body = (on_body - np.min(on_body)) / (np.max(on_body) - np.min(on_body))
    return on_body


def on_air_mid(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    on_air_mid = distance_from_joint_pcd_medoid(sim_results, body, obj_mesh)
    on_air_mid = (on_air_mid - np.min(on_air_mid)) / (np.max(on_air_mid) - np.min(on_air_mid))
    return on_air_mid


def on_air_far(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    on_air_far = -distance_from_initial_joint_position(sim_results, body, obj_mesh)
    on_air_far = (on_air_far - np.min(on_air_far)) / (np.max(on_air_far) - np.min(on_air_far))
    return on_air_far


def grasping_microgesture_region(sim_results: pd.DataFrame, body: hands.Hand, obj_mesh: Trimesh):
    distance_from_initial = distance_from_initial_joint_position(sim_results, body, obj_mesh)
    on_obj = distance_from_object(sim_results, body, obj_mesh)
    on_body = distance_from_body(sim_results, body, obj_mesh)
    on_air_mid = distance_from_joint_pcd_medoid(sim_results, body, obj_mesh)
    on_air_far = distance_from_pcd_hull(sim_results, body, obj_mesh)

    # avoid overlap with on_obj and on_body and avoid division by zero
    on_air_mid[1:] = on_air_mid[1:] / distance_from_initial[1:]
    on_air_mid[0] = on_air_mid[1:].max()
    on_air_far[1:] = on_air_far[1:] / distance_from_initial[1:]
    on_air_far[0] = on_air_far[1:].max()

    on_obj = (on_obj - np.min(on_obj)) / (np.max(on_obj) - np.min(on_obj))
    on_body = (on_body - np.min(on_body)) / (np.max(on_body) - np.min(on_body))
    on_air_mid = (on_air_mid - np.min(on_air_mid)) / (np.max(on_air_mid) - np.min(on_air_mid))
    on_air_far = (on_air_far - np.min(on_air_far)) / (np.max(on_air_far) - np.min(on_air_far))

    n = len(distance_from_initial)
    region_values = -np.ones(n)
    ranks = [np.argsort(region).tolist() for region in (on_obj, on_body, on_air_mid, on_air_far)]
    round = 0
    for i in range(n):
        idx = None
        rank = ranks[round]
        while len(rank) > 0 and (idx is None or region_values[idx] > -1):
            idx = rank.pop(0)
        if idx is not None:
            region_values[idx] = round
        round = (round + 1) % len(ranks)

    return region_values


features = [
    grasping_microgesture_region,
    distance_from_initial_joint_position,
    distance_from_joint_pcd_centroid,
    distance_from_joint_pcd_medoid,
    distance_from_pcd_hull,
    distance_from_body,
    distance_from_object
]
