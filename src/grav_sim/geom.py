from typing_extensions import Unpack

import numpy as np
from trimesh import Trimesh as Mesh
from trimesh.util import concatenate
from trimesh.collision import CollisionManager
from trimesh.voxel import VoxelGrid


def min_max_scale(vector):
    _min = np.min(vector)
    _max = np.max(vector)
    _range = _max - _min
    return (np.array(vector) - _min) / _range


def connecting_faces(faces, vertex_indexes):
    """
    Returns the faces indexes that only connect vertices which indexes are in vertex_indexes.
    """

    def _is_connecting_face(face):
        return np.all(np.isin(face, vertex_indexes))

    mask = np.apply_along_axis(_is_connecting_face, 1, faces)
    return np.where(mask)


def nearest_vertices(a, b, return_distances=False):
    """
    Returns the indexes of vertices in a that are nearest the vertices in b.
    """
    distances = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    distances = np.linalg.norm(distances, axis=-1)
    if return_distances:
        return np.argmin(distances, axis=0), distances
    return np.argmin(distances, axis=0)


def connecting_faces_of_nearest_vertices(a, b):
    distances = a.vertices[:, np.newaxis, :] - b.vertices[np.newaxis, :, :]
    distances = np.linalg.norm(distances, axis=-1)
    _nearest_vertices = np.argmin(distances, axis=1)
    return connecting_faces(b.faces, _nearest_vertices)


def nearest_neighbor(point, query_points):
    distances = np.linalg.norm(point - query_points, axis=-1)
    nn_index = distances.argmin()
    return nn_index, query_points[nn_index], distances[nn_index]


def inside_cylinder(points, bottom, top, radius):
    # Vector along the cylinder axis
    cylinder_axis = top - bottom
    cylinder_axis_length = np.linalg.norm(cylinder_axis)
    cylinder_axis_unit = cylinder_axis / cylinder_axis_length

    # Vector from p1 to the points
    vec_p1_to_points = points - bottom

    # Projection of points onto the cylinder axis
    projections = np.dot(vec_p1_to_points, cylinder_axis_unit)

    # Check if the projection lies within the cylinder height
    within_cylinder_height = (projections >= 0) & (projections <= cylinder_axis_length)

    # Closest points on the cylinder axis
    closest_points_on_axis = np.outer(projections, cylinder_axis_unit) + bottom

    # Distance from the points to the closest points on the cylinder axis
    distances_to_axis = np.linalg.norm(points - closest_points_on_axis, axis=1)

    # Check if the distance is less than or equal to the cylinder radius
    within_cylinder_radius = distances_to_axis <= radius

    # Points are inside the cylinder if they satisfy both conditions
    return within_cylinder_height & within_cylinder_radius


def split_points_with_plane(points, plane_origin, plane_normal):
    plane_normal /= np.linalg.norm(plane_normal)
    signed_dists = np.dot(points - plane_origin, plane_normal)
    positive = signed_dists > 0
    zero = np.isclose(signed_dists, 0)
    negative = signed_dists < 0
    return zero, positive, negative


def erode(mesh_a, mesh_b):
    """
    Removes from mesh_a the surface that collides with mesh_b
    """
    mesh_a = Mesh(mesh_a.vertices, mesh_a.faces)
    mesh_b = Mesh(mesh_b.vertices, mesh_b.faces)

    mesh_a.remove_duplicate_faces()
    mesh_b.remove_duplicate_faces()

    mesh_a_key = 'mesh_a'
    mesh_b_key = 'mesh_b'
    collisions = CollisionManager()
    collisions.add_object(mesh_a_key, mesh_a)
    collisions.add_object(mesh_b_key, mesh_b)

    eroded_surface = Mesh()

    is_colliding, collision_data = collisions.in_collision_internal(return_data=True)

    while is_colliding:
        # mask of colliding faces
        collision_faces = np.unique([collision.index(mesh_a_key) for collision in collision_data])
        erosion_mask = np.zeros(len(mesh_a.faces), dtype=bool)
        erosion_mask[collision_faces] = True

        # add colliding faces to eroded_surface
        copy_mesh_a = Mesh(mesh_a.vertices, mesh_a.faces)
        copy_mesh_a.update_faces(erosion_mask)
        copy_mesh_a.remove_unreferenced_vertices()
        eroded_surface += copy_mesh_a

        # remove colliding faces from mesh_a
        mesh_a.update_faces(~erosion_mask)
        mesh_a.remove_unreferenced_vertices()

        # mask largest connected component
        bodies = mesh_a.split(only_watertight=False)
        if len(bodies) > 1:
            bodies.sort(key=lambda body: body.area, reverse=True)
            largest_body, small_bodies = bodies[0], bodies[1:]
            if len(largest_body.faces) < len(mesh_a.faces):
                mesh_a = largest_body
            eroded_surface += bodies[1:]

        # repeat if mesh_a and mesh_b still collide
        collisions.remove_object(mesh_a_key)
        collisions.add_object(mesh_a_key, mesh_a)
        is_colliding, collision_data = collisions.in_collision_internal(return_data=True)

    eroded_surface = concatenate(eroded_surface)
    eroded_surface.remove_duplicate_faces()
    eroded_surface.remove_unreferenced_vertices()
    return mesh_a, eroded_surface


def voxelize(points, voxel_size=None):
    if voxel_size is None:
        diagonal = points.max(axis=0) - points.min(axis=0)
        voxel_size = np.linalg.norm(diagonal) / 10

    voxel_coordinates = np.floor(points / voxel_size).astype(np.int32)
    occupied_voxel_coordinates = np.unique(voxel_coordinates, axis=0)

    grid_min = voxel_coordinates.min(axis=0)
    grid_max = voxel_coordinates.max(axis=0)
    grid_shape = grid_max - grid_min + 1

    voxel_coordinates -= grid_min
    occupied_voxel_coordinates -= grid_min

    encoding = np.zeros(grid_shape).astype(bool)
    encoding[Unpack[occupied_voxel_coordinates.T]] = True

    transform = np.eye(4)
    transform[:3, 3] = (grid_min + .5) * voxel_size  # Translate the grid
    transform[:3, :3] *= voxel_size  # Scale the voxels by the voxel_size

    return VoxelGrid(encoding=encoding, transform=transform)


def aligning_rotation(from_vector, to_vector):
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    axis = np.cross(from_vector, to_vector)
    axis_norm = np.linalg.norm(axis)

    # If the vectors are parallel or anti-parallel, handle special cases
    if axis_norm < 1e-8:  # Vectors are parallel or anti-parallel
        if np.dot(from_vector, to_vector) > 0:  # Vectors are aligned
            return np.eye(3)
        else:  # Vectors are opposite
            orthogonal_axis = np.cross(from_vector, [1, 0, 0] if np.abs(from_vector[0]) < 1.0 else [0, 1, 0])
            orthogonal_axis /= np.linalg.norm(orthogonal_axis)
            return rodrigues_rotation_matrix(orthogonal_axis, np.pi)

    axis = axis / axis_norm  # Normalize axis
    cos_theta = np.dot(from_vector, to_vector)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return rodrigues_rotation_matrix(axis, theta)


def rodrigues_rotation_matrix(axis, theta):
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def cartesian_to_homogeneous(points):
    ones = np.ones((points.shape[0], 1))
    return np.hstack((points, ones))


def homogeneous_to_cartesian(points):
    return points[:, :-1] / points[:, -1, np.newaxis]
