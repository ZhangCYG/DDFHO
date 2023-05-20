import numpy as np

import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import PerspectiveCameras
from nnutils import mesh_utils, geom_utils
from nnutils.layers import grid_sample, grid_sample_atten
from torch.nn import init


def get_embedder(multires=10, **kwargs):
    if multires == -1:
        return nn.Identity(), kwargs.get('input_dims', 3)

    embed_kwargs = {
        'include_input': True,
        'input_dims': kwargs.get('input_dims', 3),
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'band_width': kwargs.get('band_width', 1),
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embedder_obj, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    """from https://github.com/yenchenlin/nerf-pytorch/blob/bdb012ee7a217bfd96be23060a51df7de402591e/run_nerf_helpers.py"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        band_width = self.kwargs['band_width']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = band_width * 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(band_width * 2. ** 0., band_width * 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def __call__(self, inputs):
        return self.embed(inputs)


class Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, d_out=None, dropout=0.1, pool='last'):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param pool: 'mean', 'sum', 'last' or 'none'
        '''
        super(Attention, self).__init__()
        d_out = d_model if d_out is None else d_out
        
        self.fc_q = nn.Linear(d_model, h * d_k, bias=False)
        self.fc_k = nn.Linear(d_model, h * d_k, bias=False)
        self.fc_v = nn.Linear(d_model, h * d_v, bias=False)
        self.fc_o = nn.Linear(h * d_v, d_out)
        self.dropout = nn.Dropout(dropout)
        self.pool = pool

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, seq_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param seq_mask: (b_s, n, 1)
        # :param attention_mask: Mask over attention values (b_s, 1, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1).contiguous()  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3).contiguous()  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        if attention_weights is not None:
            att = att * attention_weights
        if seq_mask is not None:
            # fill at True
            # attention_mask = (1 - torch.matmul((1 - seq_mask), (1 - seq_mask).transpose(-1, -2))).bool()  # (N*P, num_sample, num_sample)
            attention_mask = (1 - torch.matmul((1 - seq_mask[:, 0:1, :]), (1 - seq_mask[:, 1:, :]).transpose(-1, -2))).bool()  # (N*P, 1, num_sample - 1)
            attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(attention_mask, -1e10)
            att = torch.softmax(att, -1)
            att = att.masked_fill(attention_mask, 0)
        else:
            att = torch.softmax(att, -1)
        
        att = self.dropout(att)
        # print(att.shape)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, nq, self.h * self.d_v).contiguous()  # (b_s, nq, h*d_v)
        # out = self.fc_o(out)  # (b_s, nq, d_model)
        # NOTE: incremental attention
        out = queries + self.fc_o(out)  # (b_s, nq, d_model)
        
        # if seq_mask is not None:
        #     seq_mask = seq_mask[:, 0:1, :].bool()
        #     # seq_mask = seq_mask.bool()
        #     out = out.masked_fill(seq_mask, 0)
            
        # if self.pool == 'mean':
        #     # out = torch.nanmean(out, dim=1)
        #     out = out.mean(dim=1)
        # elif self.pool == 'sum':
        #     # out = torch.nansum(out, dim=1)
        #     out = out.sum(dim=1)
        # elif self.pool == 'last':
        #     # last is nan
        #     out = out[:, 0, :]
        
        return out


class PixCoord(nn.Module):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__()
        J = 16
        self.net = ImplicitNetwork(z_dim + J*6, multires=freq, 
            **cfg.DDF)
        self.attention = cfg.ATTENTION
        self.sample_gap = cfg.SAMPLE_GAP
        self.num_sample = cfg.NUM_SAMPLE
        self.multi_head = cfg.MULTI_HEAD
        self.atten_mask = cfg.ATTEN_MASK
        if self.attention:
            self.epipolar_atten = Attention(z_dim, z_dim, z_dim, self.multi_head, z_dim)

    def get_dist_joint(self, nPoints, nDirec, jsTn):
        N, P, _ = nPoints.size()
        num_j = jsTn.size(1)
        nPoints_exp = nPoints.view(N, 1, P, 3).expand(N, num_j, P, 3).reshape(N * num_j, P, 3)
        jsPoints = mesh_utils.apply_transform(nPoints_exp, jsTn.reshape(N*num_j, 4, 4)).view(N, num_j, P, 3)
        nDirec_exp = nDirec.view(N, 1, P, 3).expand(N, num_j, P, 3).reshape(N * num_j, P, 3)
        rot, trans, scale = geom_utils.homo_to_rt(jsTn.reshape(N*num_j, 4, 4))
        jsDirec = torch.bmm(rot, nDirec_exp.transpose(1, 2)).transpose(1, 2).view(N, num_j, P, 3)  # N, J, P, 3
        jsDirec = jsDirec.transpose(1, 2).reshape(N, P, num_j * 3) # N, P, J*3
        jsPoints = jsPoints.transpose(1, 2).reshape(N, P, num_j * 3) # N, P, J*3
        return torch.cat([jsPoints, jsDirec], dim=2)

    def sample_multi_z(self, xPoints, z, cTx, cam):
        N1, P, D = xPoints.size()
        N = z.size(0)
        xPoints_exp = xPoints.expand(N, P, D)

        ndcPoints = self.proj_x_ndc(xPoints_exp, cTx, cam)
        zs = mesh_utils.sample_images_at_mc_locs(z, ndcPoints)  # (N, P, D)
        return zs
    
    def sample_ray_attention_z(self, xPoints, xDirec, z, cTx, cam):
        N1, P, D = xPoints.size()
        N = z.size(0)
        xPoints_exp = xPoints.expand(N, P, D)
        endPoints = xPoints + xDirec
        endPoints_exp = endPoints.expand(N, P, D)

        ndcPoints = self.proj_x_ndc(xPoints_exp, cTx, cam)  # N, P, 2
        endNdcPoints = self.proj_x_ndc(endPoints_exp, cTx, cam)
        ndcDirec  = (endNdcPoints - ndcPoints) / torch.norm(endNdcPoints - ndcPoints, dim=2, keepdim=True) # N, P, 2
        sampleRay = ndcPoints.unsqueeze(2).repeat(1, 1, self.num_sample, 1)  # N, P, num_sample, 2
        for i in range(self.num_sample):
            sampleRay[:, :, i, :] = sampleRay[:, :, 0, :] + i * self.sample_gap * ndcDirec
        zs, mask = self.sample_images_at_mc_locs(z, sampleRay)  # (N, P, num_sample, D)
        zs = zs.masked_fill(mask, 0.)
        zs = zs.reshape(N * P, self.num_sample, -1)
        mask = mask.float()
        mask = mask.reshape(N * P, self.num_sample, -1)
        # if self.atten_mask:
        #     atten_zs = self.epipolar_atten(zs, zs, zs, mask)  # (N * P, D)
        # else:
        #     atten_zs = self.epipolar_atten(zs, zs, zs)
        # NOTE: cross attention
        if self.atten_mask:
            atten_zs = self.epipolar_atten(zs[:, 0:1, :], zs[:, 1:, :], zs[:, 1:, :], mask)  # (N * P, D)
        else:
            atten_zs = self.epipolar_atten(zs[:, 0:1, :], zs[:, 1:, :], zs[:, 1:, :])
        return atten_zs.reshape(N, P, -1)
    
    def sample_images_at_mc_locs(self, target_images, sampled_rays_xy):
        """
        Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
        this method samples the tensor `target_images` at the
        respective 2D locations.
        This function is used in order to extract the colors from
        ground truth images that correspond to the colors
        rendered using `MonteCarloRaysampler`.
        :param target_images:   (N, C, H, W)
        :param sampled_rays_xy: (N, P, n_point_on_rays, 2)
        :returns (N, P, n_point_on_rays, C)
        """
        ba = target_images.shape[0]
        dim = target_images.shape[1]
        spatial_size = sampled_rays_xy.shape[1:3]
        n_point_on_rays = sampled_rays_xy.shape[2]
        images_sampled, mask = grid_sample_atten(target_images, sampled_rays_xy.view(ba, -1, n_point_on_rays, 2))
        return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim), mask

    def proj_x_ndc(self, xPoints, cTx, cam:PerspectiveCameras):
        cPoints = mesh_utils.apply_transform(xPoints, cTx)
        ndcPoints = mesh_utils.transform_points(cPoints, cam)
        return ndcPoints[..., :2]

    def forward(self, xPointsDirec, z, hA, cTx=None, 
                cam: PerspectiveCameras=None, jsTx=None):
        N, P, _ = xPointsDirec.size()

        xPoints = xPointsDirec[:, :, :3]
        xDirec = xPointsDirec[:, :, 3:6]

        glb, local = z
        # change it (local) into attention
        if not self.attention:
            local = self.sample_multi_z(xPoints, local, cTx, cam)
        else:
            local = self.sample_ray_attention_z(xPoints, xDirec, local, cTx, cam)
        # (N, P, 3) * (N, J, 12)  --> N, J, P, 3  -> N, P, J
        dstPoints = self.get_dist_joint(xPoints, xDirec, jsTx)
        latent = self.cat_z_hA((glb, local, dstPoints), hA)
        # change direc and cat as a feature
        points = self.net.cat_z_point(xPointsDirec, latent)
        # edited net
        mask_ddf_value = self.net(points)
        mask_ddf_value = mask_ddf_value.view(N, P, 2)
        return mask_ddf_value

    # TODO: modify to fit DDF Elkonal loss
    def gradient(self, xPoints, sdf):
        """
        Args:
            x ([type]): (N, P, 3)
        Returns:
            Grad sdf_x: (N, P, 3)
        """
        xPoints.requires_grad_(True)
        y = sdf(xPoints)  # (N, P, 1)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xPoints,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # only care about xyz dim
        # gradients = gradients[..., -3:]

        # only care about sdf within cube
        xyz = xPoints[..., -3:]
        within_cube = torch.all(torch.abs(xyz) < 1, dim=-1, keepdim=True).float()  # (NP, )
        gradients = within_cube * gradients + (1 - within_cube) * 1 / np.sqrt(gradients.size(-1))

        # if self.cfg.GRAD == 'clip':
        #     mask = (y.abs() <= 0.1).float()
        #     gradients = mask * gradients
        # else:
        #     pass
        return gradients

    def cat_z_hA(self, z, hA):
        glb, local, dst_points = z
        out = torch.cat([(glb.unsqueeze(1) + local), dst_points], -1)
        return out


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            latent_dim,
            feature_vector_size=0,
            d_in=6,
            d_out=1,
            DIMS=[ 512, 512, 512, 512, 512, 512, 512, 512 ], 
            GEOMETRIC_INIT=False,
            bias=1.0,
            SKIP_IN=(4, ),
            weight_norm=True,
            multires=10,
            th=True,
            depth_scale=1.,
            **kwargs
    ):
        self.xyz_dim = d_in
        super().__init__()
        dims = [d_in + latent_dim] + list(DIMS) + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch + latent_dim

        self.num_layers = len(dims)
        self.skip_in = SKIP_IN
        self.mask_out = (3, )
        self.layers = nn.ModuleDict()
        self.mask_out_layer = nn.Sigmoid()
        self.depth_scale = depth_scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            elif l + 1 in self.mask_out:
                out_dim = dims[l + 1] + 1  # additional mask
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if GEOMETRIC_INIT:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    # torch.nn.init.constant_(lin.weight[:, 0:latent_dim], 0.0)
                    torch.nn.init.constant_(lin.weight[:, latent_dim+3:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # setattr(self, "lin" + str(l), lin)
            self.layers.add_module("lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        if th:
            # self.th = nn.Tanh()
            self.relu = nn.ReLU()
        else:
            # self.th = nn.Identity()
            self.relu = nn.Sigmoid()
        
    def forward(self, input, compute_grad=False):

        xyz = input[:, -self.xyz_dim:]  # xyz + direc
        latent = input[:, :-self.xyz_dim]

        if self.embed_fn is not None:
            xyz = self.embed_fn(xyz)
        input = torch.cat([latent, xyz], dim=1)
        x = input

        for l in range(0, self.num_layers - 1):
            # lin = getattr(self, "lin" + str(l))
            lin = self.layers["lin" + str(l)]

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            
            if l in self.mask_out:
                mask = self.mask_out_layer(x[:, -1:])
                x = x[:, :-1]

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        # x = self.th(x)
        x = self.relu(x) * self.depth_scale
        # within_cube = torch.all(torch.abs(xyz[:, :3]) <= 1, dim=-1, keepdim=True).float()  # (NP, )
        # # with torch.no_grad():
        #     # apprx_dist= (torch.norm(xyz, dim=-1, keepdim=True) - 1).clamp(min=.3)
        # apprx_dist = .3
        # apprx_mask = .5
        # x = within_cube * x + (1 - within_cube) * (apprx_dist)
        # mask = within_cube * mask + (1 - within_cube) * (apprx_mask)
        return torch.cat([mask, x], dim=-1)
    
    # TODO: modify to fit DDF Elkonal loss
    def gradient(self, x, sdf=None):
        """
        :param x: (sumP, D?+3)
        :return: (sumP, 1, 3)
        """
        x.requires_grad_(True)
        if sdf is None:
            y = self.forward(x)[:, :1]
        else:
            y = sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        # only care about xyz dim
        gradients = gradients[..., -3:]

        # only care about sdf within cube
        xyz = x[..., -self.xyz_dim:]
        within_cube = torch.all(torch.abs(xyz[:, :3]) < 1, dim=-1, keepdim=True).float()  # (NP, )
        gradients = within_cube * gradients + (1 - within_cube) * 1 / np.sqrt(gradients.size(-1))
        return gradients.unsqueeze(1)

    def cat_z_point(self, points, z):
        """
        :param points: (N, P, 3)
        :param z: (N, (P, ), D)
        :return: (NP, D+6)
        """
        if z.ndim == 3:
            N, P, D = z.size()
            return torch.cat([z, points], dim=-1).reshape(N*P, D+6)
        N, D = z.size()
        if points.ndim == 2:
            points = points.unsqueeze(0)
        NP, P, _ = points.size()
        assert N == NP

        z_p = torch.cat([z.unsqueeze(1).repeat(1, P, 1), points], dim=-1)
        z_p = z_p.reshape(N * P, D + 6)
        return z_p


class PixObj(PixCoord):
    def __init__(self, cfg, z_dim, hA_dim, freq):
        super().__init__(cfg, z_dim, hA_dim, freq)
        J = 16
        self.net = ImplicitNetwork(z_dim, multires=freq, 
            **cfg.DDF)
    
    def cat_z_hA(self, z, hA):
        glb, local, _ = z 
        glb = glb.unsqueeze(1)
        return glb + local


def build_net(cfg, z_dim=None):
    if z_dim is None:
        z_dim = cfg.Z_DIM
    if cfg.DEC == 'obj':
        dec = PixObj(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ)
    else:
        dec = PixCoord(cfg, z_dim, cfg.THETA_DIM, cfg.FREQ)
    return dec

