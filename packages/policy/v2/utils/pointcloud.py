import torch
import numpy as np
import math
import open3d as o3d
from scipy.stats import special_ortho_group

from utils.sphere import sample_from_Uniform

def get_partial_pc_via_depth(mesh, cam_poses, num_pcd_points=3000, get_color=True, plot=False, visualize_pc_with_mesh=False, render_mesh=False):
    depth_im_width = 640 * 2
    depth_im_height = 1000
    FOV_H = 75 # degree
    FOV_V = 65 # degree
    fx = 498.83063258
    fy = 498.83063258
    cx = 0.5
    cy = 0.5
    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic.set_intrinsics(depth_im_width, depth_im_height, fx, fy, depth_im_width * cx - 0.5, depth_im_height * cy - 0.5)
    diameter = 1.5 * np.linalg.norm(mesh.get_oriented_bounding_box().extent)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=depth_im_width, height=depth_im_height, visible=render_mesh)
    vis.add_geometry(mesh)
    vis.get_render_option().load_from_json("utils/RenderOption.json")

    cam_poses = np.asarray(cam_poses) # cam pos from obj (desired orientation, distance not fixed)
    cam_poses = cam_poses.reshape((-1,3))
    num_samples = cam_poses.shape[0]
    partial_pcds = []
    partial_pcds_numpy = []
    voxel_size = []
    for sample in range(num_samples):
        cam_up = np.array([0, 0, 1]) # upward to cam
        theta = np.arccos(np.dot(cam_up, cam_poses[sample, :]) / np.linalg.norm(cam_poses[sample, :]))
        if abs(theta) < 1e-4 or abs(math.pi - theta) < 1e-4: # near z axis
            cam_up = np.array([0, 1, 0])
        cam_target = mesh.get_center() # obj pos
        cam_pos = mesh.get_center() + diameter * cam_poses[sample, :] / np.linalg.norm(cam_poses[sample, :]) # cam pos global, diameter
        
        cam_look = cam_pos - cam_target
        cam_look = cam_look / np.linalg.norm(cam_look) # obj to cam direction

        cam_right = np.cross(cam_up, cam_look)
        cam_right = cam_right / np.linalg.norm(cam_right) # rightward to cam

        cam_up = np.cross(cam_look, cam_right)
        cam_up = cam_up / np.linalg.norm(cam_up) # upward to cam (redefined)

        cam_R = np.array([cam_right, - cam_up, - cam_look]) # cam orientation
        diameter_zoom = diameter
        cnt = 1
        while True:
            cam_t = - np.dot(cam_R, cam_pos)
            
            cam_extrinsic_matrix = np.identity(4)
            cam_extrinsic_matrix[:3, :3] = cam_R
            cam_extrinsic_matrix[:3, 3] = cam_t

            camera.extrinsic = cam_extrinsic_matrix

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(camera)
            
            vis.poll_events()
            vis.update_renderer()

            depth = vis.capture_depth_float_buffer()
            RGB =  vis.capture_screen_float_buffer()
            RGB = o3d.geometry.Image((np.asarray(RGB) * 255).astype(np.uint8))
            # asdfasdf = np.asarray(depth)
            # print(asdfasdf.shape)
            # print(camera.extrinsic)
            # plt.imshow(asdfasdf)
            # plt.show()
            if get_color:
                RGBD = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, depth, convert_rgb_to_intensity=False, depth_scale=1, depth_trunc=3)
                partial_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBD, camera.intrinsic, cam_extrinsic_matrix)
                partial_pcd.paint_uniform_color([0, 0, 0.9])
            else:	
                partial_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera.intrinsic, cam_extrinsic_matrix)
                partial_pcd.paint_uniform_color([0, 0, 0.9])

            num_orig_pcd_pnts = np.asarray(partial_pcd.points).shape[0]
            if num_orig_pcd_pnts > num_pcd_points:
                break

            cnt += 1
            print('zooming')
            diameter_zoom = (1.5 - cnt * 0.1) * np.linalg.norm(mesh.get_oriented_bounding_box().extent)

            cam_pos = mesh.get_center() + diameter_zoom * cam_poses[sample, :] / np.linalg.norm(cam_poses[sample, :])
        # downsample
        v_size = 0.5
        v_min = 0
        v_max = 1
        v_last_met_requirements = None
        cnt = 0
        for trial in range(100):
            num_tmp_pcd_pnts = np.asarray(partial_pcd.voxel_down_sample(v_size).points).shape[0]
            if num_tmp_pcd_pnts - num_pcd_points >= 0 and num_tmp_pcd_pnts - num_pcd_points < 10:
                break 
            if num_tmp_pcd_pnts > num_pcd_points:
                v_last_met_requirements = v_size
                v_min = v_size

            if num_tmp_pcd_pnts < num_pcd_points:
                v_max = v_size
            v_size = (v_min + v_max) / 2
            if trial == 99 and num_tmp_pcd_pnts > num_pcd_points:
                if v_last_met_requirements is not None:
                    v_size = v_last_met_requirements
                else:
                    v_size = v_min
            cnt += 1
        # print('cnt', cnt)
        partial_pcd = partial_pcd.voxel_down_sample(v_size)
        partial_pcd = partial_pcd.select_by_index([*range(num_pcd_points)])
        partial_pcds.append(partial_pcd)
        partial_pcds_numpy.append(np.asarray(partial_pcd.points))
        voxel_size.append(v_size)

    vis.destroy_window()

    if plot is True:
        for sample in range(num_samples):
            if visualize_pc_with_mesh:
                o3d.visualization.draw_geometries([mesh, partial_pcds[sample]])
            else:
                o3d.visualization.draw_geometries([partial_pcds[sample]])

    return partial_pcds_numpy, voxel_size

def get_cam_pnts(samples=1, random=False, plot=False):

    cam_points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        z = 1 - (i / float(samples - 1)) * 2  # z goes from 1 to -1
        radius = math.sqrt(1 - z * z)  # radius at z

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        y = - math.sin(theta) * radius

        cam_points.append((x, y, z))

    cam_points = np.asarray(cam_points)
    
    if random is True:
        rand_rot = special_ortho_group.rvs(3)
        cam_points = np.transpose(np.matmul(rand_rot, np.transpose(cam_points)))
    
    if plot is True:
        cam_pcd = o3d.geometry.PointCloud(points = o3d.utility.Vector3dVector(cam_points))
        cam_pcd.paint_uniform_color([1, 0, 0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius = 0.99, resolution = 50)
        o3d.visualization.draw_geometries([sphere, cam_pcd])

    return cam_points


def partialpointclouds(mesh, view_num = 10, viewpoint = None, sample_ppc_num = 500, visualize = False, visualize_mesh = True):

    if viewpoint == None:
        cam_pnts = get_cam_pnts(view_num, plot=False)
    elif viewpoint == 'uniform':
        cam_pnts = sample_from_Uniform(view_num).squeeze(0)
    else:
        cam_pnts = viewpoint.reshape(-1, 3)
        view_num = cam_pnts.shape[0]

    partial_pcds, voxel_size = get_partial_pc_via_depth(mesh, cam_pnts, get_color=True, render_mesh=False, plot=visualize, 
                                                        visualize_pc_with_mesh=visualize_mesh, num_pcd_points=sample_ppc_num)

    return partial_pcds