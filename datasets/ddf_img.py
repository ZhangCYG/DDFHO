from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import pytorch3d.ops as op_3d
from nnutils.hand_utils import ManopthWrapper, get_nTh
from nnutils import mesh_utils, geom_utils, image_utils


class DdfImg(nn.Module):
    """DDF Wrapper of datasets"""
    def __init__(self, cfg, dataset, is_train, data_dir='../data/', base_idx=0):
        super().__init__()
        print(cfg)
        self.cfg = cfg
        self.dataset = dataset
        self.train = is_train
        
        self.anno = {
            'index': [],  # per grasp
            'cad_index': [],
            'hA': [],   # torch.Float (45? )
            'hTo': [],  # torch Float (4, 4)
            'cTh': [],  # torch.Float (4, 4)
        }

        self.base_idx = base_idx
        self.data_dir = data_dir

        self.subsample = cfg.DB.NUM_POINTS
        self.hand_wrapper = ManopthWrapper().to('cpu')

        self.transform = transforms.Compose([
            transforms.ColorJitter(.4,.4,.4),
            transforms.ToTensor(),
        ]) if self.train else transforms.ToTensor()

    def preload_anno(self):
        self.dataset.preload_anno(self.anno.keys())
        self.dataset.check_anno()
        for key in self.anno:
            self.anno[key] = self.dataset.anno[key]
        self.obj2mesh = self.dataset.obj2mesh
        self.map = self.dataset.map

    def __len__(self):
        return len(self.anno['index'])

    def __getitem__(self, idx):
        sample = {}
        
        idx = self.map[idx] if self.map is not None else idx
        # load DDF
        cad_idx = self.anno['cad_index'][idx]
        filename = self.dataset.get_ddf_files(cad_idx)
        sample['cad_name'] = filename
        oPos_ddf = unpack_ddf_samples(filename, None)
        hTo = torch.FloatTensor(self.anno['hTo'][idx])
        hA = torch.FloatTensor(self.anno['hA'][idx])
        # add noise
        # hA = hA + torch.randn(hA.shape) * 0.1
        nTh = get_nTh(self.hand_wrapper, hA[None], self.cfg.DB.RADIUS)[0]

        # handle direction
        oDdf = self.sample_points(oPos_ddf, self.subsample)
        sample['oDdf'] = oDdf.clone()

        # hDdf = self.norm_points_ddf(oDdf, hTo)
        # sample['hDdf'] = hDdf.clone()

        nPos_ddf = self.norm_points_ddf(oPos_ddf, nTh @ hTo)
        nDdf = self.sample_unit_cube(nPos_ddf, self.subsample)
        # print(torch.norm(nDdf[:, 3:6], dim=1))
        sample['nDdf'] = nDdf.clone()

        # load pointcloud
        mesh = self.obj2mesh[cad_idx]
        if self.cfg.MODEL.BATCH_SIZE == 1:
            sample['mesh'] = mesh
        xyz, color = op_3d.sample_points_from_meshes(mesh, self.subsample, return_textures=True)
        sample['oObj'] = torch.cat([xyz, color], dim=-1)[0]  # (1, P, 6)
        # hObj = torch.cat([
        #         mesh_utils.apply_transform(xyz, hTo[None]),
        #         color,
        #     ], dim=-1)[0]
        # sample['hObj'] = hObj

        xyz = mesh_utils.apply_transform(xyz, (nTh @ hTo)[None])
        nObj = torch.cat([xyz, color], dim=-1)[0]  # (1, P, 6)
        nObj = self.sample_unit_cube(nObj, self.subsample)
        sample['nObj'] = nObj

        sample['hA'] = self.rdn_hA(hA)
        sample['nTh'] = geom_utils.matrix_to_se3(nTh)
        sample['hTo'] = geom_utils.matrix_to_se3(hTo)
        sample['indices'] = idx + self.base_idx

        # add crop?? 
        sample['cTh'] = geom_utils.matrix_to_se3(self.anno['cTh'][idx].squeeze(0))
        
        sample['bbox'] = self.get_bbox(idx)
        sample['cam_f'], sample['cam_p'] = self.get_f_p(idx, sample['bbox'])
        sample['image'] = self.get_image(idx, sample['bbox'])
        sample['obj_mask'] = self.get_obj_mask(idx, sample['bbox'])

        sample['index'] = self.get_index(idx)
        # import ipdb
        # ipdb.set_trace()
        return sample   

    def rdn_hA(self, hA):
        if self.train:
            hA = hA + (torch.rand([45]) * self.cfg.DB.JIT_ART * 2 - self.cfg.DB.JIT_ART)
        return hA
    
    def norm_points_ddf(self, obj, nTh):
        """
        :param obj: (P, 9)
        :param nTh: (4, 4)
        :return:
        """
        # D = 9

        xyz, direc, mask_ddf_sym = obj[None].split([3, 3, 3], dim=-1)  # (N, Q, 3)
        nXyz = mesh_utils.apply_transform(xyz, nTh[None])  # (N, Q, 3)

        # nDirec = mesh_utils.apply_transform(direc, nTh[None])

        rot, trans, scale = geom_utils.homo_to_rt(nTh)  # (N, 3)

        # nDirec = nDirec / scale[..., 0:1, None]
        # nDirec = nDirec - trans

        nDirec = torch.mm(rot, direc.squeeze(0).t()).t()
        # ddf
        # print(nDirec.shape)
        # print(torch.norm(nDirec, dim=1))
        # print(scale)
        # import ipdb
        # ipdb.set_trace()

        mask_ddf_sym[:, :, 1] = mask_ddf_sym[:, :, 1] * scale[..., 0:1, None]
        nObj = torch.cat([nXyz, nDirec.unsqueeze(0), mask_ddf_sym], dim=-1)
        return nObj[0]

    def sample_points(self, points, num_points):
        """

        Args:
            points ([type]): (P, D)
        Returns:
            sampled points: (num_points, D)
        """
        P, D = points.size()
        ones = torch.ones([P])
        inds = torch.multinomial(ones, num_points, replacement=True).unsqueeze(-1)  # (P, 1)
        points = torch.gather(points, 0, inds.repeat(1, D))
        return points

    def sample_unit_cube(self, hObj, num_points, r=1):
        """
        Args:
            points (P, 9): Description
            num_points ( ): Description
            r (int, optional): Description
        
        Returns:
            sampled points: (num_points, 9)
        """
        D = hObj.size(-1)
        points = hObj[..., :3]
        prob = (torch.sum((torch.abs(points) < r), dim=-1) == 3).float()
        if prob.sum() == 0:
            prob = prob + 1
            print('oops')
        inds = torch.multinomial(prob, num_points, replacement=True).unsqueeze(-1)  # (P, 1)

        handle = torch.gather(hObj, 0, inds.repeat(1, D))
        return handle

    def get_index(self, idx):
        index =  self.anno['index'][idx]
        if isinstance(index, tuple) or isinstance(index, list):

            index = '/'.join(index)
        return index

    def get_bbox(self, idx):
        bbox =  self.dataset.get_bbox(idx)  # in scale of pixel torch.floattensor 
        bbox = image_utils.square_bbox(bbox)
        bbox = self.jitter_bbox(bbox)
        return bbox

    def get_f_p(self, idx, bbox):
        cam_intr = self.dataset.get_cam(idx)  # with pixel?? in canvas
        cam_intr = image_utils.crop_cam_intr(cam_intr, bbox, 1)
        f, p = image_utils.screen_intr_to_ndc_fp(cam_intr, 1, 1)
        f, p = self.jitter_fp(f, p) 
        return f, p

    def get_image(self, idx, bbox):
        image = np.array(self.dataset.get_image(self.anno['index'][idx]))
        image = image_utils.crop_resize(image, bbox, return_np=False)
        return self.transform(image) * 2 - 1
        
    def get_obj_mask(self, idx, bbox):
        obj_mask = np.array(self.dataset.get_obj_mask(self.anno['index'][idx]))
        # obj_mask = np.array(self.anno['obj_mask'][idx])
        obj_mask = image_utils.crop_resize(obj_mask, bbox,return_np=False)
        return (self.transform(obj_mask) > 0).float()

    def jitter_bbox(self, bbox):
        if self.train:
            bbox = image_utils.jitter_bbox(bbox, 
                self.cfg.DB.JIT_SCALE, self.cfg.DB.JIT_TRANS)
        return bbox
    
    def jitter_fp(self, f, p):
        if self.train:
            stddev_p = self.cfg.DB.JIT_P / 224 * 2
            dp = torch.rand_like(p) * stddev_p * 2 - stddev_p
            p += dp
        return f, p


def unpack_ddf_samples(filename, subsample=None):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz).to(torch.float32)
    if subsample is None:
        return pos_tensor

    random_pos = (torch.rand(subsample) * pos_tensor.shape[0]).long()
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    samples = sample_pos

    return samples


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def vis_db():
    from pytorch3d.renderer import PerspectiveCameras
    from config.args_config_ddf import default_argument_parser, setup_cfg
    from nnutils import model_utils, hand_utils
    args = default_argument_parser().parse_args()
    cfg = setup_cfg(args)

    from . import build_dataloader_ddf
    from nnutils import image_utils
    jitter = False
    data_loader = build_dataloader_ddf(cfg, 'train', jitter, shuffle=False, bs=1)
    device = 'cuda:0'
    hand_wrapper = hand_utils.ManopthWrapper().to(device)

    for i, data in enumerate(data_loader):

        data = model_utils.to_cuda(data)

        image_utils.save_images(data['image'], osp.join(save_dir, '%d_gt_%d' % (i, jitter)), scale=True)

        N, P, _ = data['nDdf'].size()
        cameras = PerspectiveCameras(data['cam_f'], data['cam_p'], device=device)

        nDdf = data['nDdf'][..., P // 2:, :3]
        cTn = geom_utils.compose_se3(data['cTh'], geom_utils.inverse_rt(data['nTh']))
        nObj = mesh_utils.pc_to_cubic_meshes(nDdf)
        cObj = mesh_utils.apply_transform(nObj, cTn)
        cHand, _ = hand_wrapper(data['cTh'], data['hA'])
        cHoi = mesh_utils.join_scene([cHand, cObj])
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_cObj' % i))
        image = mesh_utils.render_mesh(cHoi, cameras)
        image_utils.save_images(image['image'], osp.join(save_dir, '%d_cObj_%s_%d' % (i, cfg.DB.NAME, jitter)), 
            bg=data['image'], mask=image['mask'], scale=True)
        
        nDdf = data['nDdf'][..., P // 2:, :]
        cTn = geom_utils.compose_se3(data['cTh'], geom_utils.inverse_rt(data['nTh']))
        nPc = mesh_utils.ddf_to_pc(nDdf)
        # print(nPc.shape)
        nObj = mesh_utils.pc_to_cubic_meshes(nPc)
        cObj = mesh_utils.apply_transform(nObj, cTn)
        cHand, _ = hand_wrapper(data['cTh'], data['hA'])
        cHoi = mesh_utils.join_scene([cHand, cObj])
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_cObj_ddf' % i))
        image = mesh_utils.render_mesh(cHoi, cameras)
        image_utils.save_images(image['image'], osp.join(save_dir, '%d_cObj_ddf_%s_%d' % (i, cfg.DB.NAME, jitter)), 
            bg=data['image'], mask=image['mask'], scale=True)
        
        N, P, _ = data['nObj'].size()
        nDdf = data['nObj'][..., P // 2:, :3]
        cTn = geom_utils.compose_se3(data['cTh'], geom_utils.inverse_rt(data['nTh']))
        nObj = mesh_utils.pc_to_cubic_meshes(nDdf)
        cObj = mesh_utils.apply_transform(nObj, cTn)
        cHand, _ = hand_wrapper(data['cTh'], data['hA'])
        cHoi = mesh_utils.join_scene([cHand, cObj])
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_cObj_gt' % i))
        image = mesh_utils.render_mesh(cHoi, cameras)
        image_utils.save_images(image['image'], osp.join(save_dir, '%d_cObj_gt_%s_%d' % (i, cfg.DB.NAME, jitter)), 
            bg=data['image'], mask=image['mask'], scale=True)

        N, P, _ = data['nDdf'].size()
        nDdf = data['nDdf'][..., P // 2:, :]
        nPc = mesh_utils.ddf_to_pc(nDdf)
        nObj = mesh_utils.pc_to_cubic_meshes(nPc)
        image_list = mesh_utils.render_geom_rot(nObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_nObj_ddf' % i))

        zeros = torch.zeros([N, 3], device=device)
        hHand, _ = hand_wrapper(None, data['hA'], zeros, mode='inner')
        xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, data['nTh']), nObj])
        image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_nObj_ddf_hand' % i))

        N, P, _ = data['nObj'].size()
        oDdf = data['nObj'][..., P // 2:, :3]
        oObj = mesh_utils.pc_to_cubic_meshes(oDdf)
        image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_nObj_mesh' % i))

        N, P, _ = data['nDdf'].size()
        oDdf = data['nDdf'][..., P // 2:, :3]
        oObj = mesh_utils.pc_to_cubic_meshes(oDdf)
        image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_nObj_self' % i))

        # print(data['cad_name'])
        N, P, _ = data['oObj'].size()
        oDdf = data['oObj'][:, :, :3]
        oObj = mesh_utils.pc_to_cubic_meshes(oDdf)
        image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_oObj_mesh' % i))

        N, P, _ = data['oDdf'].size()
        oDdf = data['oDdf']
        oPc = mesh_utils.ddf_to_pc(oDdf)
        # oPc = torch.from_numpy(oPc).to(device)
        # mask = (oDdf[:, :, 6] > 0.5).unsqueeze(2).repeat(9, dim=2)
        # oDdf_val = oDdf[mask]
        # print(oPc - oDdf_val[:, :, :3])
        oObj = mesh_utils.pc_to_cubic_meshes(oPc)
        image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_oObj_ddf' % i))

        # N, P, _ = data['oDdf'].size()
        # oLoc = data['oDdf'][:, :, 9:]
        # mask = (oDdf[:, :, 6] > 0.5)
        # oObj = mesh_utils.pc_to_cubic_meshes(oLoc[mask])
        # image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        # image_utils.save_gif(image_list, osp.join(save_dir, '%d_oObj_loc' % i))

        N, P, _ = data['oDdf'].size()
        oDdf = data['oDdf'][:, :, :3]
        oObj = mesh_utils.pc_to_cubic_meshes(oDdf)
        image_list = mesh_utils.render_geom_rot(oObj, scale_geom=True)
        image_utils.save_gif(image_list, osp.join(save_dir, '%d_oObj_self' % i))

        print(torch.norm(nDdf[:, :, 3:6], dim=2))

        if i >= 2 :
            break


if __name__ == '__main__':
    import os.path as osp
    # save_dir = 'out/demo/vis_ddf_ho3d/'
    save_dir = 'out/demo/vis_ddf'
    # save_dir = 'out/demo/vis_ddf_mow'
    vis_db()
