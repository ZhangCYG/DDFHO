# generate the folder of mesh hand
# if mesh obj doesn't exist, this can also be used to generate mesh_obj
import os
import argparse
import trimesh
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')
from ddfutils.DDF_sampling import *
from ddfutils.DDF_generation import *
import pickle
import torch
import pytorch3d


def ddf_to_pc(ddf):
    # ddf: [p, 9]
    points = ddf[:, :3]
    direc = ddf[:, 3:6]
    mask = ddf[:, 6] > 0.5
    depth = ddf[:, 7]
    pc = points[mask] + direc[mask] * np.expand_dims(depth[mask], axis=1).repeat(3, axis=1)
    return pc


parser = argparse.ArgumentParser()
parser.add_argument('--output_root', default='processed_data/mow/', help="Path to dataset root")
args = parser.parse_args()

res_path = args.output_root

obj_sample_num_list = [20000, 7000, 7000, 7000, 7000, 512]
#total: 48000

cache_path = 'data/cache/mow_train_mesh.pkl'
cache_mesh = pickle.load(open(cache_path, 'rb'))

for key in tqdm(cache_mesh.keys()):
    print(key)
    obj_mesh_py = cache_mesh[key]

    if os.path.exists(res_path + key + '.npy'):
        try:
            npy = np.load(res_path + key + '.npy')
            print(npy.shape)
            continue
        except:
            print('Empty!')
    
    # center, radius = trimesh.nsphere.minimum_nsphere(obj_mesh.vertices)
    # obj_mesh.apply_translation(-center)
    # obj_mesh.apply_scale(1.0 / radius)

    verts = obj_mesh_py._verts_list[0].numpy()
    faces = obj_mesh_py._faces_list[0].numpy()
    obj_mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # sample different type of rays
    # in total 6 categories
    # object sample
    s1_start_obj, s1_end_obj = sample_within_sphere_start_end_uniformly(obj_sample_num_list[0])
    s2_start_obj, s2_end_obj = sample_start_uniform_within_sphere_end_mesh(obj_mesh, obj_sample_num_list[1])
    s3_start_obj, s3_end_obj = sample_start_noise_end_mesh_tangent(obj_mesh, obj_sample_num_list[2])
    s4_start_obj, s4_end_obj = sample_end_mesh_tangent_noise(obj_mesh, 0.1, obj_sample_num_list[3])
    s5_start_obj, s5_end_obj = sample_start_on_sphere_end_mesh(obj_mesh, obj_sample_num_list[4])
    s_start_obj = np.concatenate([s1_start_obj, s2_start_obj, s3_start_obj, s4_start_obj, s5_start_obj], axis=0)
    s_end_obj = np.concatenate([s1_end_obj, s2_end_obj, s3_end_obj, s4_end_obj, s5_end_obj], axis=0)

    print("Sample Finished!")

    # sample ddf
    # obj
    obj_p, obj_d, obj_m, obj_depth, locations = batch_direct_ddf_generation(s_start_obj, s_end_obj, obj_mesh)
    print("DDF Finished!")

    # symmetry detection
    plane_normal_list = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    flag = 0
    for plane_normal in plane_normal_list:
        s_sym = sample_symmetry_point(plane_normal, obj_sample_num_list[5])
        # sym obj sampling
        sym_flag, sym_p, sym_d, sym_m, sym_depth = sym_mesh_detection(s_sym, plane_normal, obj_mesh)
        flag += 1
        if sym_flag:
            break
        
    print("Sym Detection Finished!")
    # if not sym_flag:
    #     sym_p = np.zeros([obj_sample_num_list[-1], 3])
    #     sym_d = np.zeros([obj_sample_num_list[-1]], 3)
    #     sym_m = np.zeros([obj_sample_num_list[-1]])
    #     sym_depth = np.zeros([obj_sample_num_list[-1]])
    
    # NOTE: add scale? NO?
    # NOTE: in mm ? NO

    # x y z d1 d2 d3 mask depth sym
    # sym = 0 means non - sym object
    # sym = 1, 2, 3 means sym axis

    res = np.concatenate([obj_p, obj_d, np.expand_dims(obj_m, axis=1), np.expand_dims(obj_depth, axis=1)], axis=1)
    if sym_flag:
        if flag == 1:
            res = np.concatenate([res, np.ones((res.shape[0], 1))], axis=1)
        elif flag == 2:
            res = np.concatenate([res, np.full((res.shape[0], 1), 2)], axis=1)
        elif flag == 3:
            res = np.concatenate([res, np.full((res.shape[0], 1), 3)], axis=1)
    else:
        res = np.concatenate([res, np.zeros((res.shape[0], 1))], axis=1)
    print(res.shape)
    # x y z d1 d2 d3 mask depth sym
    print(res[:, 0].min())
    print(res[:, 0].max())
    print(res[:, 1].min())
    print(res[:, 1].max())
    print(res[:, 2].min())
    print(res[:, 2].max())
    # print(res[:, 3].min())
    # print(res[:, 3].max())
    print(np.linalg.norm(res[:, 3:6], axis=1).mean())
    # print(np.linalg.norm(res[:, 3:6], axis=1).min())
    # print(res[:, 4].min())
    # print(res[:, 4].max())
    # print(res[:, 5].min())
    # print(res[:, 5].max())
    # print(res[:, 6].min())
    print(np.sum(res[:, 6]))
    print(res[:, 7].min())
    print(res[:, 7].max())
    print(res[:, 7].mean())
    print(res[:, 8].mean())
    # print(res[:, 8].max())

    # check
    mask = res[:, 6] > 0.5
    pc = ddf_to_pc(res)
    print('Check: ', np.mean(np.abs(pc - locations[mask])))

    # for debug
    # res = np.concatenate([res, locations], axis=1)

    np.save(res_path + key + '.npy', res)

print("Successfully generate DDF samples!")
