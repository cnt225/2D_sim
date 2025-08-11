import torch
import numpy as np

from urdfpy import URDF
import fcl, trimesh

def collision_check(data, SE3, type='mesh'):
    if type == 'mesh':
        unit1 = 0.066 #* 8 # 0.56
        unit2 = 0.041 #* 8 # 0.32
        unit3 = 0.046 #* 8 # 0.4
        pbase = torch.Tensor([0, 0, 0, 1]).reshape(1, -1)
        pcenter = torch.Tensor([0, 0, unit1, 1]).reshape(1, -1)
        pleft = torch.Tensor([unit2, 0, unit1, 1]).reshape(1, -1)
        pright = torch.Tensor([-unit2, 0, unit1, 1]).reshape(1, -1)
        plefttip = torch.Tensor([unit2, 0, unit1+unit3, 1]).reshape(1, -1)
        prighttip = torch.Tensor([-unit2, 0, unit1+unit3, 1]).reshape(1, -1)
        hand = torch.cat([pbase, pcenter, pleft, pright, plefttip, prighttip], dim=0).to(SE3)
        hand = torch.einsum('ij, kj -> ik', SE3, hand).cpu()

        phandx = [hand[0,4], hand[0,2], hand[0,1], hand[0,0], hand[0,1], hand[0,3], hand[0,5]]
        phandy = [hand[1,4], hand[1,2], hand[1,1], hand[1,0], hand[1,1], hand[1,3], hand[1,5]]
        phandz = [hand[2,4], hand[2,2], hand[2,1], hand[2,0], hand[2,1], hand[2,3], hand[2,5]]

        vertices = torch.Tensor(data.vertices)
        faces = data.triangles
        for i in range(len(phandx)-1):
            point1 = torch.Tensor([phandx[i], phandy[i], phandz[i]])
            point2 = torch.Tensor([phandx[i+1], phandy[i+1], phandz[i+1]])
            
            pointA = vertices[faces][:,0]
            pointB = vertices[faces][:,1]
            pointC = vertices[faces][:,2]

            vec1 = pointB - pointA
            vec2 = pointC - pointA
            n = torch.cross(vec1, vec2, dim=1)
            n = torch.nn.functional.normalize(n)
            d = torch.einsum('ni, ni->n', n, vertices[faces][:,0])

            p = point1
            dir = point2-point1
            t = (d - torch.einsum('ni, i->n', n, p)) / torch.einsum('ni, i->n', n, dir)
            PonT = p.reshape(-1, 3).repeat(len(n), 1) + torch.einsum('n, ni->ni', t, dir.reshape(-1, 3).repeat(len(n), 1))
            
            condition1 = (t>=0) * (t<=1)
            condition2 = torch.einsum('ni, ni->n', torch.cross(pointB - pointA, PonT - pointA, dim=1), n)>=0
            condition3 = torch.einsum('ni, ni->n', torch.cross(pointC - pointB, PonT - pointB, dim=1), n)>=0
            condition4 = torch.einsum('ni, ni->n', torch.cross(pointA - pointC, PonT - pointC, dim=1), n)>=0
            if torch.sum(condition1 * condition2 * condition3 * condition4)>0:
                # print('collision'+str(i))
                return True
    return False

def check_grasp_collision(mesh, grasp, gripper_mesh='simplified'):
    """_summary_

    Args:
        mesh (open3d TriangleMesh): mesh of the target object
        grasp (np.array): (4, 4) SE(3) matrix of grasp
        gripper_mesh (str): 'simplified' or 'original', 2.5x fast calculation if choose 'simplified'
    """
    
    objV = np.array(mesh.vertices)
    objT = np.array(mesh.triangles)
    
    objmodel = fcl.BVHModel()
    objmodel.beginModel(len(objV), len(objT))
    objmodel.addSubModel(objV, objT)
    objmodel.endModel()
    
    object = fcl.CollisionObject(objmodel, fcl.Transform())
    
    gripper = URDF.load('./assets/robot/urdfs/panda_gripper_fixpath.urdf')
    if gripper_mesh == 'simplified':
        linkSE3 = gripper.collision_trimesh_fk(cfg={'panda_finger_joint1': 0.04})
    elif gripper_mesh == 'original':
        linkSE3 = gripper.visual_trimesh_fk(cfg={'panda_finger_joint1': 0.04})
    gripper_meshes = list(linkSE3.keys())
    
    gripper_collision_manager = fcl.DynamicAABBTreeCollisionManager()
    for trimesh in gripper_meshes:
        tmpV = np.array(trimesh.vertices)
        tmpT = np.array(trimesh.faces)
        
        tmpSE3 = linkSE3[trimesh]
        tmpSE3 = grasp @ np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) @ tmpSE3
        
        tmpshape = fcl.BVHModel()
        tmpshape.beginModel(len(tmpV), len(tmpT))
        tmpshape.addSubModel(tmpV, tmpT)
        tmpshape.endModel()
        tmpobj = fcl.CollisionObject(tmpshape, fcl.Transform(tmpSE3[:3, :3], tmpSE3[:3, 3]))
        
        gripper_collision_manager.registerObject(tmpobj)
        
    req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
    rdata = fcl.CollisionData(request = req)

    gripper_collision_manager.collide(object, rdata, fcl.defaultCollisionCallback)
    
    return rdata.result.is_collision