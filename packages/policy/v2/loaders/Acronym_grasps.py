import h5py
import os
import open3d as o3d
import pickle
import numpy as np


class AcronymGrasps:
    def __init__(self, root_dir, file_dir):
        self.root_dir = root_dir

        data = h5py.File(os.path.join(root_dir, file_dir), 'r')

        self.mesh_dir = data['object/file'][()].decode('utf-8')
        self.mesh_scale = data['object/scale'][()]

        self.grasps = data['grasps/transforms'][()]
        self.success = data['grasps/qualities/flex/object_in_gripper'][()]
        self.good_grasps = self.grasps[self.success == 1]
        self.bad_grasps = self.grasps[self.success == 0]

    def load_mesh(self):
        mesh_dir = os.path.join(self.root_dir, self.mesh_dir)

        self.mesh = o3d.io.read_triangle_mesh(mesh_dir)

        self.mesh.scale(self.mesh_scale, center=[0, 0, 0])

        return self.mesh
    
    def get_sdf(self, num_coords):
        sdf_dir = os.path.join(self.root_dir, self.mesh_dir.replace('meshes', 'sdf').replace('.obj', '.json'))

        with open(sdf_dir, 'rb') as f:
            sdf_dict = pickle.load(f)

        xyz = (sdf_dict['xyz'] + sdf_dict['loc']) * sdf_dict['scale'] * self.mesh_scale
        shuffle_idxs = np.random.permutation(len(xyz))
        xyz = xyz[shuffle_idxs[:num_coords]]
        sdf = sdf_dict['sdf'][shuffle_idxs[:num_coords]] * sdf_dict['scale'] * self.mesh_scale

        return xyz, sdf
