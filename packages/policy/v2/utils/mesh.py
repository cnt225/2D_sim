from copy import deepcopy
import open3d as o3d
import numpy as np


def scene_generator(mesh_list, Ts_grasp_list):
    scene_list = []

    for mesh, Ts_grasp in zip(mesh_list, Ts_grasp_list):
        scene = deepcopy(mesh)

        for T_grasp in Ts_grasp:
            mesh_base_1 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.066, resolution=6, split=1)
            T_base_1 = np.eye(4)
            T_base_1[:3, 3] = [0, 0, 0.033]
            mesh_base_1.transform(T_base_1)

            mesh_base_2 = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.082, resolution=6, split=1)
            T_base_2 = np.eye(4)
            T_base_2[:3, :3] = mesh_base_2.get_rotation_matrix_from_xyz([0, np.pi/2, 0])
            T_base_2[:3, 3] = [0, 0, 0.066]
            mesh_base_2.transform(T_base_2)

            mesh_left_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.046, resolution=6, split=1)
            T_left_finger = np.eye(4)
            T_left_finger[:3, 3] = [-0.041, 0, 0.089]
            mesh_left_finger.transform(T_left_finger)

            mesh_right_finger = o3d.geometry.TriangleMesh.create_cylinder(radius=0.002, height=0.046, resolution=6, split=1)
            T_right_finger = np.eye(4)
            T_right_finger[:3, 3] = [0.041, 0, 0.089]
            mesh_right_finger.transform(T_right_finger)

            mesh_gripper = mesh_base_1 + mesh_base_2 + mesh_left_finger + mesh_right_finger

            mesh_gripper.transform(T_grasp)

            scene += mesh_gripper

        scene.compute_vertex_normals()
        scene.paint_uniform_color([0.5, 0.5, 0.5])

        scene_list += [scene]

    return scene_list


def meshes_to_numpy(mesh_list):
    total_vertices = []
    total_triangles = []
    total_colors = []

    max_num_vertices = 0
    max_num_triangles = 0

    for idx, mesh in enumerate(mesh_list):
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        colors = 255 * np.asarray(mesh.vertex_colors)

        # append for batch
        total_vertices += [vertices]
        total_triangles += [triangles]
        total_colors += [colors]

        # find the maximum dimension
        if len(vertices) > max_num_vertices:
            max_num_vertices = len(vertices)
        if len(triangles) > max_num_triangles:
            max_num_triangles = len(triangles)

    # matching dimension between batches for tensorboard
    for idx in range(len(mesh_list)):
        diff_num_vertices = max_num_vertices - len(total_vertices[idx])
        diff_num_triangles = max_num_triangles - len(total_triangles[idx])

        total_vertices[idx] = np.concatenate((total_vertices[idx], np.zeros((diff_num_vertices, 3))), axis=0)
        total_triangles[idx] = np.concatenate((total_triangles[idx], np.zeros((diff_num_triangles, 3))), axis=0)
        total_colors[idx] = np.concatenate((total_colors[idx], np.zeros((diff_num_vertices, 3))), axis=0)

    return np.asarray(total_vertices), np.asarray(total_triangles), np.asarray(total_colors)
