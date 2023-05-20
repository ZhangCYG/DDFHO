from __future__ import print_function
import json
import os
import os.path as osp
import pickle
import numpy as np
import tqdm
from PIL import Image

import torch
from pytorch3d.transforms.transform3d import Rotate, Scale, Translate
from nnutils.hand_utils import ManopthWrapper, cvt_axisang_t_i2o
from datasets.base_data import BaseData, minmax, proj3d

from nnutils import mesh_utils, geom_utils, image_utils


def apply_trans(rot, t, s, device='cpu'):
    trans = Translate(torch.tensor(t).reshape((1, 3)), device=device)
    rot = Rotate(torch.tensor(rot).reshape((1, 3, 3)), device=device)
    scale = Scale(s).to(device)

    chain = rot.compose(trans, scale)
    mat = chain.get_matrix().transpose(-1, -2)
    return mat


class MOW(BaseData):
    def __init__(self, cfg, dataset: str, split='val', is_train=True,
                 data_dir='../data/', cache=None):
        data_dir = osp.join(data_dir, 'mow')
        super().__init__(cfg, 'mow', split, is_train, data_dir)
        # self.num_points = cfg.DB.NUM_POINTS
        self.cache = cache if cache is not None else self.cfg.DB.CACHE
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],
            'cTh': [],
            'hTo': [],
            'bbox': [],
            'cam': [],
        }
        self.suf = dataset[len('rhoi'):]
        if split == 'val':
            self.split = 'test'
        self.set_dir = osp.join(self.data_dir, '{}.lst')
        self.cache_file = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s.pkl' % (dataset, self.split))
        self.cache_mesh = osp.join(osp.dirname(self.data_dir), 'cache', '%s_%s_mesh.pkl' % (dataset, self.split))
        self.mask_dir = osp.join(self.data_dir, 'results/{0}/{0}_mask.png')
        self.image_dir = osp.join(self.data_dir, 'images/{0}.jpg')
        self.shape_dir = osp.join(self.data_dir, 'results/{0}/{0}_norm.obj')

        self.hand_wrapper = ManopthWrapper().to('cpu')
    
    def check_anno(self):
        for id, cad_idx in enumerate(self.anno['cad_index']):
            try:
                filename = self.get_ddf_files(cad_idx)
                npz = np.load(filename)
                if npz.shape[0] != 48000:
                    self.anno['index'][id] = -1
            except:
                self.anno['index'][id] = -1
        new_index = []
        new_cad = []
        new_ha = []
        new_hto = []
        new_cth = []
        new_bbox = []
        new_cam = []
        for id, cad_idx in enumerate(self.anno['index']):
            if self.anno['index'][id] != -1:
                new_index.append(self.anno['index'][id])
                new_cad.append(self.anno['cad_index'][id])
                new_ha.append(self.anno['hA'][id])
                new_hto.append(self.anno['hTo'][id])
                new_cth.append(self.anno['cTh'][id])
                new_bbox.append(self.anno['bbox'][id])
                new_cam.append(self.anno['cam'][id])
        self.anno['index'] = new_index
        self.anno['cad_index'] = new_cad
        self.anno['hA'] = new_ha
        self.anno['hTo'] = new_hto
        self.anno['cTh'] = new_cth
        self.anno['bbox'] = new_bbox
        self.anno['cam'] = new_cam

    def preload_anno(self, load_keys=...):
        if self.cache and osp.exists(self.cache_file):
            print('!! Load from cache !!', self.cache_file)
            self.anno = pickle.load(open(self.cache_file, 'rb'))
        else:
            if 'mini' in self.suf:            
                index_list = [line.strip() for line in open(self.set_dir.format('all'))]
                index_list = ['gardening_v_qLis0UwnJkc_frame000220'] #, 'study_v_aJobyfOfMj0_frame000254']
            else:
                index_list = [line.strip() for line in open(self.set_dir.format(self.split))]

            index_list = set(index_list)
            with open(osp.join(self.data_dir, 'poses.json')) as fp:
                anno_list = json.load(fp)

            for i, anno in enumerate(anno_list):
                if len(self.suf) > 0 and len(self) >= int(self.suf[len('mini'):]):
                    break
                index = anno['image_id']
                if index not in index_list:
                    continue

                self.anno['index'].append(index)
                self.anno['cad_index'].append(index)

                mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0)
                transl = torch.tensor(anno['trans']).unsqueeze(0)
                pose, global_orient = mano_pose[:, 3:], mano_pose[:, :3]
                pose = pose + self.hand_wrapper.hand_mean

                wrTh = geom_utils.axis_angle_t_to_matrix(*cvt_axisang_t_i2o(global_orient, transl))
                wTwr = apply_trans(anno['hand_R'], anno['hand_t'], anno['hand_s'])
                wTh = wTwr @ wrTh

                oToo = geom_utils.rt_to_homo(torch.eye(3) * 8)
                wTo = apply_trans(anno['R'], anno['t'], anno['s'])
                wTo = wTo @ oToo
                hTo = geom_utils.inverse_rt(mat=wTh, return_mat=True) @ wTo

                # todo: inverse
                image = Image.open(self.image_dir.format(index))
                W, H = image.size
                cam_intr = torch.FloatTensor([
                    [2, 0, 0],
                    [0, 2, 0],
                    [0, 0, 1],
                ])
                cam_intr = image_utils.ndc_to_screen_intr(cam_intr, H, W)

                _, cJoints = self.hand_wrapper(geom_utils.matrix_to_se3(wTh), pose)
                pad = 0.8
                bbox2d = image_utils.square_bbox(minmax(proj3d(cJoints, cam_intr))[0], pad)

                self.anno['bbox'].append(bbox2d)
                self.anno['cam'].append(cam_intr)
                self.anno['cTh'].append(wTh[0])
                self.anno['hTo'].append(hTo[0])
                self.anno['hA'].append(pose[0])
                # self.anno['image'].append(image)
                
            os.makedirs(osp.dirname(self.cache_file), exist_ok=True)
            print('save cache')
            pickle.dump(self.anno, open(self.cache_file, 'wb'))

        self.preload_mesh()

    def preload_mesh(self):
        if self.cache and osp.exists(self.cache_mesh):
            print('!! Load from cache !!')
            self.obj2mesh = pickle.load(open(self.cache_mesh, 'rb'))
        else:
            self.obj2mesh = {}
            print('load mesh')
            for i, cls_id in tqdm.tqdm(enumerate(self.anno['cad_index']), total=len(self.anno['cad_index'])):
                key = cls_id
                if key not in self.obj2mesh:
                    fname = self.shape_dir.format(cls_id)
                    self.obj2mesh[key] = mesh_utils.load_mesh(fname, scale_verts=1)
            print('save cache')
            pickle.dump(self.obj2mesh, open(self.cache_mesh, 'wb'))

    def get_bbox(self, idx):
        return self.anno['bbox'][idx]
    
    def get_cam(self, idx):
        return self.anno['cam'][idx]

    def get_obj_mask(self, index):
        """fake mask"""
        image = np.array(Image.open(self.image_dir.format(index)))
        H, W, _= image.shape
        mask = Image.fromarray(np.ones([H, W]).astype(np.uint8) * 255)
        return mask

    def get_image(self, index):
        return Image.open(self.image_dir.format(index))        
