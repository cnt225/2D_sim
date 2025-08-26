import torch
import os
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation

from loaders.Acronym_grasps import AcronymGrasps


OBJ_TYPES = ['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil', 
               'Plate', 'ScrewDriver', 'WineBottle', 'Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear', 
               'Book', 'Books', 'Camera', 'CerealBox', 'Cookie', 'Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting', 
               'PillBottle', 'Plant', 'PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes', 
               'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan', 'Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper', 
               'ToyFigure', 'Wallet', 'WineGlass', 'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 
               'Tank', 'Truck', 'USBStick']


class AcronymDataset4RCFM(torch.utils.data.Dataset):
    def __init__(self, split, obj_types=None, num_point_cloud=1000, num_grasps=1000, scale=8, augmentation=True, **kwargs):
        if obj_types is None:
            self.obj_types = OBJ_TYPES
        else:
            self.obj_types = obj_types

        root_dir = 'datasets/Acronym'

        self.num_point_cloud = num_point_cloud
        self.split = split
        self.augmentation = augmentation
        self.scale = scale

        # gather data
        self.mesh_list_train = []
        self.Ts_grasp_list_train = []

        self.pc_list_class = []
        self.Ts_grasp_list_class = []
        self.mesh_list_class = []
        self.obj_id_list_class = []
        self.quat_list_class = []
        self.mean_list_class = []

        for obj_type in self.obj_types:
            grasp_files = sorted(os.listdir(os.path.join(root_dir, 'grasps', obj_type)))

            # split data
            num_val_data = num_test_data = len(grasp_files) // 5
            num_train_data = len(grasp_files) - num_val_data - num_test_data

            if split == 'train':
                obj_ids = range(num_train_data)
            elif split == 'valid':
                obj_ids = range(num_train_data, num_train_data+num_val_data)
            elif split == 'test':
                obj_ids = range(num_train_data+num_val_data, num_train_data+num_val_data+num_test_data)

            # obj_ids = np.load(f'datasets/Acronym/splits/{obj_type}/idxs_{split}.npy')

            pc_list_object = []
            Ts_grasp_list_object = []
            mesh_list_object = []
            obj_id_list_object = []
            quat_list_object = []
            mean_list_object = []

            for obj_id in obj_ids:
                grasp_file = grasp_files[obj_id]

                # load grasp data
                grasp_obj = AcronymGrasps(root_dir=root_dir, file_dir=os.path.join('grasps', obj_type, grasp_file))

                if len(grasp_obj.good_grasps) == 0:
                    continue

                # load mesh
                mesh = grasp_obj.load_mesh()

                # get grasp poses
                Ts_grasp = grasp_obj.good_grasps

                # rescale
                mesh.scale(scale, center=[0, 0, 0])
                Ts_grasp[:, :3, 3] *= scale

                # get point cloud
                pc = np.asarray(mesh.sample_points_uniformly(num_point_cloud).points)

                # translate
                mean = pc.mean(axis=0)
                mesh.translate(-mean)
                pc -= mean
                Ts_grasp[:, :3, 3] -= mean

                if (split == 'train') or ((split == 'valid' or 'test') and not augmentation):
                    mesh_ = [mesh]
                    pc_ = [pc]
                    Ts_grasp_ = [Ts_grasp]
                    mean_ = [mean]
                    quat_ = [[0, 0, 0, 1]]

                else:
                    # expand dataset
                    Rs = np.load('loaders/rotations.npy')
                    Ts = np.tile(np.eye(4), (len(Rs), 1, 1))
                    Ts[:, :3, :3] = Rs

                    mesh_ = []

                    for R in Rs:
                        mesh_rot = deepcopy(mesh)
                        mesh_rot.rotate(R, center=[0, 0, 0])

                        mesh_ += [mesh_rot]

                    pc_ = np.einsum('bij,nj->bni', Rs, pc)
                    Ts_grasp_ = np.einsum('bij,njk->bnik', Ts, Ts_grasp)
                    mean_ = np.einsum('bij,j->bi', Rs, mean)
                    quat_ = Rotation.from_matrix(Rs).as_quat().tolist()

                pc_list_rot = []
                Ts_grasp_list_rot = []
                mesh_list_rot = []
                quat_list_rot = []
                mean_list_rot = []

                for mesh, pc, Ts_grasp, quat, mean in zip(mesh_, pc_, Ts_grasp_, quat_, mean_):
                    # split grasp poses
                    Ts_grasp_split = [Ts_grasp_.numpy().copy() for Ts_grasp_ in torch.from_numpy(Ts_grasp).split(num_grasps)]
                    Ts_grasp_split[-1] = np.concatenate([Ts_grasp_split[-1], np.zeros([num_grasps - len(Ts_grasp_split[-1]), 4, 4])])

                    # append data to dataset
                    self.mesh_list_train += [mesh] * len(Ts_grasp_split)
                    self.Ts_grasp_list_train += Ts_grasp_split

                    pc_list_rot += [pc]
                    Ts_grasp_list_rot += [Ts_grasp]
                    mesh_list_rot += [mesh]
                    quat_list_rot += [quat]
                    mean_list_rot += [mean]

                pc_list_object += [pc_list_rot]
                Ts_grasp_list_object += [Ts_grasp_list_rot]
                mesh_list_object += [mesh_list_rot]
                obj_id_list_object += [obj_id]
                quat_list_object += [quat_list_rot]
                mean_list_object += [mean_list_rot]

            if len(Ts_grasp_list_object):
                self.pc_list_class += [pc_list_object]
                self.Ts_grasp_list_class += [Ts_grasp_list_object]
                self.mesh_list_class += [mesh_list_object]
                self.obj_id_list_class += [obj_id_list_object]
                self.quat_list_class += [quat_list_object]
                self.mean_list_class += [mean_list_object]
            else:
                self.obj_types.remove(obj_type)

    def __len__(self):
        return len(self.Ts_grasp_list_train)

    def __getitem__(self, index):
        # load mesh and grasp poses
        mesh = self.mesh_list_train[index]
        Ts_grasp = self.Ts_grasp_list_train[index].copy()

        # sample point cloud
        pc = np.asarray(mesh.sample_points_uniformly(self.num_point_cloud).points)

        # translate
        mean = pc.mean(axis=0)
        pc -= mean
        Ts_grasp[:, :3, 3] -= mean

        if self.split == 'train' and self.augmentation:
            # # randomly rotate
            # R = Rotation.random().as_matrix()

            # randomly rotate along z axis
            degree = np.random.rand() * 2 * np.pi
            R = Rotation.from_rotvec(degree * np.array([0, 0, 1])).as_matrix()

            T = np.eye(4)
            T[:3, :3] = R

            pc = (R @ pc.T).T
            Ts_grasp = T @ Ts_grasp

        return {'pc': torch.Tensor(pc), 'Ts_grasp': torch.Tensor(Ts_grasp)}
