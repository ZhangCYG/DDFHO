import functools
from typing import Any, List

import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch3d.renderer.cameras import PerspectiveCameras

from config.args_config_ddf import default_argument_parser, setup_cfg
from nnutils.logger import MyLogger
from datasets import build_dataloader_ddf
from models import dec_ddf, enc
from nnutils.hand_utils import ManopthWrapper
from nnutils import geom_utils, mesh_utils


def get_hTx(frame, batch):
    hTn = geom_utils.inverse_rt(batch['nTh'])
    hTx = hTn
    return hTx


def get_jsTx(hand_wrapper, hA, hTx):
    """
    Args:
        hand_wrapper ([type]): [description]
        hA ([type]): [description]
        hTx ([type]): se3
    Returns: 
        (N, 4, 4)
    """
    hTjs = hand_wrapper.pose_to_transform(hA, False) 
    N, num_j, _, _ = hTjs.size()
    jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
    hTx = geom_utils.se3_to_matrix(hTx
            ).unsqueeze(1).repeat(1, num_j, 1, 1)
    jsTx = jsTh @ hTx
    return jsTx



class IHoiDdf(pl.LightningModule):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        # self.hparams = cfg
        self.hparams.update(cfg)
        self.cfg = cfg

        self.dec = dec_ddf.build_net(cfg.MODEL)  # ddf
        self.enc = enc.build_net(cfg.MODEL.ENC, cfg)  # ImageSpEnc(cfg, out_dim=cfg.MODEL.Z_DIM, layer=cfg.MODEL.ENC_RESO, modality=cfg.DB.INPUT)
        self.hand_wrapper = ManopthWrapper()

        self.minT = 0.
        self.maxT = cfg.LOSS.DDF_MINMAX
        self.ddf_key = '%sDdf' % cfg.MODEL.FRAME[0]
        self.obj_key = '%sObj' % cfg.MODEL.FRAME[0]
        self.metric = 'val'
        self._train_loader = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.SOLVER.BASE_LR)

    def train_dataloader(self):
        if self._train_loader is None:
            loader = build_dataloader_ddf(self.cfg, 'train')
            self._train_loader = loader
        return self._train_loader

    def val_dataloader(self):
        test = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        val_dataloader = build_dataloader_ddf(self.cfg, 'test', is_train=False, name=test)
        trainval_dataloader = build_dataloader_ddf(self.cfg, 'train', is_train=True, 
            shuffle=False, bs=min(8, self.cfg.MODEL.BATCH_SIZE), name=self.cfg.DB.NAME)
        return [val_dataloader, trainval_dataloader]
    
    def test_dataloader(self):
        test = self.cfg.DB.NAME if self.cfg.DB.TESTNAME == '' else self.cfg.DB.TESTNAME
        val_dataloader = build_dataloader_ddf(self.cfg, self.cfg.TEST.SET, is_train=False, name=test)
        return [val_dataloader, ]

    def get_jsTx(self, hA, hTx):
        hTjs = self.hand_wrapper.pose_to_transform(hA, False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                  ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp        
        return jsTx

    def ddf(self, hA, ddf_hA_jsTx, hTx):
        ddf = functools.partial(ddf_hA_jsTx, hA=hA, jsTx=self.get_jsTx(hA, hTx))
        return ddf

    def forward(self, batch):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        xTh = geom_utils.inverse_rt(hTx)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=batch['image'].device)

        with torch.enable_grad():
            ddf_hA_jsTx = functools.partial(self.dec, 
                z=image_feat, cTx=cTx, cam=cameras)
            ddf_hA = functools.partial(self.ddf, ddf_hA_jsTx=ddf_hA_jsTx, hTx=hTx)
            ddf = ddf_hA(batch['hA'])

        out = {
            'ddf': ddf,
            'ddf_hA': ddf_hA,
            'hTx': hTx,
            'xTh': xTh,
        }
        return out
    
    def training_step(self, batch, batch_idx):        
        losses, out = self.step(batch, batch_idx)
        losses = {'train_' + e: v for e,v in losses.items()}
        # loss
        if self.trainer.is_global_zero:
            self.log_dict(losses)

            # print every
            if self.global_step % self.hparams.TRAIN.PRINT_EVERY == 0:
                self.logger.print(self.global_step, self.current_epoch, losses, losses['train_loss'])
        return losses['train_loss']

    def test_step(self, *args):
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError

        prefix = '%d_%d' % (dataloader_idx, batch_idx)
        losses, out = self.step(batch, 0)
        if batch_idx % 10 == 0:
            # for sanity check
            self.vis_step(out, batch, prefix)
        f_res = self.quant_step(out, batch)

        return f_res

    def test_epoch_end(self, outputs: List[Any], save_dir=None) -> None:
        save_dir = self.logger.local_dir if save_dir is None else save_dir
        mean_list = mesh_utils.test_end_fscore(outputs, save_dir)
        
    def validation_step(self, *args):
        return args

    def validation_step_end(self, batch_parts_outputs):
        args = batch_parts_outputs
        if len(args) == 3:
            batch, batch_idx, dataloader_idx = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
                dataloader_idx = dataloader_idx[0]
        elif len(args) == 2:
            batch, batch_idx, = args
            if not isinstance(batch_idx, int):
                batch_idx = batch_idx[0]
            dataloader_idx = 0
        else:
            raise NotImplementedError
        prefix = '%d_%d' % (dataloader_idx, batch_idx)

        losses, out = self.step(batch, 0)
        losses = {'val_' + e: v for e,v in losses.items()}
        # val loss
        self.log_dict(losses, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self.vis_step(out, batch, prefix)
            self.quant_step(out, batch)
        return losses

    def quant_step(self, out, batch, ddf=None):
        device = batch['cam_f'].device
        N = batch['cam_f'].size(0)

        if ddf is None:
            camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
            cTx = geom_utils.compose_se3(batch['cTh'], get_hTx(self.cfg.MODEL.FRAME, batch))
            # normal space, joint space jsTn, image space 
            ddf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], 
                jsTx=out['jsTx'], cTx=cTx, cam=camera)
        # use pc for metrics
        xObj = mesh_utils.batch_ddf_to_pc(ddf, N, batch[self.ddf_key])

        th_list = [.5/100, 1/100,]
        gt_pc = batch[self.obj_key][..., :3]

        xObj = xObj.to(device)
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        hObj = mesh_utils.apply_transform(xObj, hTx) 
        hGt = mesh_utils.apply_transform(gt_pc, hTx)
        f_res = mesh_utils.fscore(hObj, hGt, num_samples=gt_pc.size(1), th=th_list)

        # f_res, cd = mesh_utils.fscore(xObj, gt_pc, num_samples=gt_pc.size(1), th=th_list)
        for th, th_f in zip(th_list, f_res[:-1]):
            self.log('f-%d' % (th*100), np.mean(th_f), sync_dist=True)
        self.log('cd', np.mean(f_res[-1]), sync_dist=True)
        return  [batch['indices'].tolist()] + f_res

    def vis_input(self, out, batch, prefix):
        N = len(batch['nObj'])
        device = batch['nObj'].device

        self.logger.save_images(self.global_step, batch['image'], '%s_image' % prefix)

        zeros = torch.zeros([N, 3], device=device)
        hHand, _ = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/hand' % (self.global_step, prefix)), hHand)

        # hDdf = mesh_utils.pc_to_cubic_meshes(mesh_utils.apply_transform(
        #             batch[self.ddf_key][:, P//2:, :3], get_hTx(self.cfg.MODEL.FRAME, batch)
        #     ))
        # hHoi = mesh_utils.join_scene([hHand, hDdf])
        
        # cHoi = mesh_utils.apply_transform(hHoi, batch['cTh'])
        # cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        # image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        # self.logger.save_gif(self.global_step, image_list, '%s_inp' % prefix)
        hObj = mesh_utils.pc_to_cubic_meshes(mesh_utils.apply_transform(batch[self.obj_key][:, batch[self.obj_key].size(1)//2:, :3], get_hTx(self.cfg.MODEL.FRAME, batch)))
        hHoi = mesh_utils.join_scene([hHand, hObj])
        
        cHoi = mesh_utils.apply_transform(hHoi, batch['cTh'])
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=cameras)
        self.logger.save_gif(self.global_step, image_list, '%s_gt' % prefix)
        
        return {'hHand': hHand}
    
    def vis_output(self, out, batch, prefix, cache={}):
        N = len(batch['nObj'])
        device = batch['nObj'].device
        zeros = torch.zeros([N, 3], device=device)
        hHand, hJoints = self.hand_wrapper(None, batch['hA'], zeros, mode='inner')
        cJoints = mesh_utils.apply_transform(hJoints, batch['cTh'])
        cache['hHand'] = hHand

        # output mesh
        camera = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=device)
        cTx = geom_utils.compose_se3(batch['cTh'], get_hTx(self.cfg.MODEL.FRAME, batch))
        # normal space, joint space jsTn, image space 
        ddf = functools.partial(self.dec, z=out['z'], hA=batch['hA'], 
            jsTx=out['jsTx'], cTx=cTx, cam=camera)

        xPc = mesh_utils.batch_ddf_to_pc(ddf, N, batch[self.ddf_key])
        xObj = mesh_utils.pc_to_cubic_meshes(xPc.to(device))
        cache['xMesh'] = xObj
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        hObj = mesh_utils.apply_transform(xObj, hTx)
        mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/obj' % (self.global_step, prefix)), hObj)
        xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx)), xObj])
        image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        self.logger.save_gif(self.global_step, image_list, '%s_xHoi' % prefix)

        cHoi = mesh_utils.apply_transform(xHoi, cTx)
        image = mesh_utils.render_mesh(cHoi, camera)
        self.logger.save_images(self.global_step, image['image'], '%s_cam_mesh' % prefix,
            bg=batch['image'], mask=image['mask'])
        image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=camera,
            xyz=cJoints[:, 5], out_size=512)
        self.logger.save_gif(self.global_step, image_list, '%s_cHoi' % prefix)

        # TODO: for visulization may convert to meshes directly
        # xObj = mesh_utils.batch_ddf_to_pc_to_mesh(ddf, N, batch[self.ddf_key])
        # xObj = xObj.to(device)
        # cache['xMesh'] = xObj
        # hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        # hObj = mesh_utils.apply_transform(xObj, hTx)
        # mesh_utils.dump_meshes(osp.join(self.logger.local_dir, '%d_%s/obj' % (self.global_step, prefix)), hObj)
        # xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx)), xObj])
        # image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        # self.logger.save_gif(self.global_step, image_list, '%s_xHoi_mesh' % prefix)

        # cHoi = mesh_utils.apply_transform(xHoi, cTx)
        # image = mesh_utils.render_mesh(cHoi, camera)
        # self.logger.save_images(self.global_step, image['image'], '%s_cam_mesh' % prefix,
        #     bg=batch['image'], mask=image['mask'])
        # image_list = mesh_utils.render_geom_rot(cHoi, view_centric=True, cameras=camera,
        #     xyz=cJoints[:, 5], out_size=512)
        # self.logger.save_gif(self.global_step, image_list, '%s_cHoi_mesh' % prefix)

        xPc = mesh_utils.ddf_to_pc(batch[self.ddf_key])
        xObj = mesh_utils.pc_to_cubic_meshes(xPc)
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx)), xObj])
        image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        self.logger.save_gif(self.global_step, image_list, '%s_xHoi_gt_ddf' % prefix)

        xPc = batch[self.ddf_key][:, :, :3]
        xObj = mesh_utils.pc_to_cubic_meshes(xPc)
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        xHoi = mesh_utils.join_scene([mesh_utils.apply_transform(hHand, geom_utils.inverse_rt(hTx)), xObj])
        image_list = mesh_utils.render_geom_rot(xHoi, scale_geom=True)
        self.logger.save_gif(self.global_step, image_list, '%s_xHoi_sam_ddf' % prefix)

        return cache

    def vis_step(self, out, batch, prefix):
        cache = self.vis_input(out, batch, prefix)
        cache = self.vis_output(out, batch, prefix, cache)
        return cache

    def step(self, batch, batch_idx):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        
        xXyzDirec = batch[self.ddf_key][..., :6]
        hTx = get_hTx(self.cfg.MODEL.FRAME, batch)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=xXyzDirec.device)

        hTjs = self.hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx = geom_utils.se3_to_matrix(hTx
                ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx

        pred_ddf = self.dec(xXyzDirec, image_feat, batch['hA'], cTx, cameras, jsTx=jsTx, )
        
        # pred_ddf is (..., 2)
        out = {self.ddf_key: pred_ddf, 'z': image_feat, 'jsTx': jsTx}

        loss, losses = 0., {}
        cfg = self.cfg.LOSS

        # TODO: loss
        # xXyz = batch[self.ddf_key][..., :3]
        # ndcPoints = mesh_utils.transform_points(mesh_utils.apply_transform(xXyz, cTx), cameras)
        recon_loss, bce_loss = self.ddf_loss(pred_ddf, batch[self.ddf_key][..., 6:8], cfg.RECON, cfg.ENFORCE_MINMAX, )
        
        loss = loss + recon_loss + bce_loss
        losses['recon'] = recon_loss
        losses['bce'] = bce_loss
        losses['loss'] = loss

        return losses, out

    def ddf_loss(self, ddf_pred, ddf_gt, wgt=1, minmax=False, ):
        # TODO: add symmetry
        # TODO: add Elkonal

        mode = self.cfg.LOSS.OFFSCREEN  # [gt, out, idc]
        if mode == 'gt':
            pass
        # elif mode == 'out':
        #     mask = torch.all(ndcPoints <= 1, dim=-1, keepdim=True) &\
        #          torch.all(ndcPoints >= -1, dim=-1, keepdim=True)
        #     value = self.maxT if self.cfg.MODEL.OCC == 'ddf' else 1
        #     ddf_gt = mask * ddf_gt + (~mask) * value
        # elif mode == 'idc':
        #     mask = torch.any(ndcPoints <= 1, dim=-1, keepdim=True) & \
        #         torch.any(ndcPoints >= -1, dim=-1, keepdim=True)
        #     ddf_pred = ddf_pred * mask  # the idc region to zero
        #     ddf_gt = ddf_gt * mask
        # else:
        #     raise NotImplementedError

        mask_p = ddf_pred[:, :, :1]
        ddf_p = ddf_pred[:, :, 1:2]
        mask_g = ddf_gt[:, :, :1]
        ddf_g = ddf_gt[:, :, 1:2]

        bce_loss = F.binary_cross_entropy(mask_p, mask_g)

        # if minmax or self.current_epoch >= self.cfg.TRAIN.EPOCH // 2:
        #     ddf_p = torch.clamp(ddf_p, self.minT, self.maxT)
        #     ddf_g = torch.clamp(ddf_g, self.minT, self.maxT)

        recon_loss = wgt * F.l1_loss(ddf_p * mask_g, ddf_g * mask_g)
        # recon_loss = wgt * F.mse_loss(ddf_p * mask_g, ddf_g * mask_g)

        # assert (ddf_g < 0).equal(mask_g == 0)
        # assert torch.count_nonzero(torch.lt(ddf_g * mask_g, 0.)) == 0
        
        return recon_loss, bce_loss


def main(cfg, args):
    pl.seed_everything(cfg.SEED)
    
    model = IHoiDdf(cfg)
    if args.ckpt is not None:
        print('load from', args.ckpt)
        model = model.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False)

    # instantiate model
    if args.eval:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=True,
                        )
        trainer = pl.Trainer(gpus='0,',
                             default_root_dir=cfg.MODEL_PATH,
                             logger=logger,
                            #  resume_from_checkpoint=args.ckpt,
                             )
        print(cfg.MODEL_PATH, trainer.weights_save_path, args.ckpt)

        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        logger = MyLogger(save_dir=cfg.OUTPUT_DIR,
                        name=os.path.dirname(cfg.MODEL_SIG),
                        version=os.path.basename(cfg.MODEL_SIG),
                        subfolder=cfg.TEST.DIR,
                        resume=True,
                        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor='val_loss/dataloader_idx_0',
            mode='min',
            save_last=True,
        )
        lr_monitor = LearningRateMonitor()

        # every_iter = len(model.train_dataloader())
        max_epoch = cfg.TRAIN.EPOCH # max(cfg.TRAIN.EPOCH, cfg.TRAIN.ITERS // every_iter)
        trainer = pl.Trainer(
                             gpus=-1,
                             accelerator='dp',
                             num_sanity_val_steps=1,
                             limit_val_batches=2,
                             check_val_every_n_epoch=cfg.TRAIN.EVAL_EVERY,
                             default_root_dir=cfg.MODEL_PATH,
                             logger=logger,
                             max_epochs=max_epoch,
                             callbacks=[checkpoint_callback, lr_monitor],            
                             )
        trainer.fit(model)





if __name__ == '__main__':
    arg_parser = default_argument_parser()
    # arg_parser = slurm_utils.add_slurm_args(arg_parser)
    args = arg_parser.parse_args()
    
    cfg = setup_cfg(args)
    save_dir = os.path.dirname(cfg.MODEL_PATH)
    main(cfg, args)
    # slurm_utils.slurm_wrapper(args, save_dir, main, {'args': args, 'cfg': cfg}, resubmit=False)
