import torch
from torch import nn
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum
from .modules import (
    Encoder,
    CamEncode,
    BevEncode,
    BevPost,
    SceneUnder,
    Embedder_lr1,
    Embedder_lr2,
    Embedder_f1,
    Embedder_f2,
    Predictor,
    TemporalConcat1x1,
)


class BEV_TXT(nn.Module):
    def __init__(self, bsize, grid_conf, data_aug_conf, outC):
        super(BEV_TXT, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.bsize = bsize
        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        self.encoder = Encoder()
        self.sceneunder = SceneUnder()

        self.embeder_f1 = Embedder_f1(in_channels=256, out_channels=32)
        self.embeder_f2 = Embedder_f2(out_channels=40)
        self.embeder_lr1 = Embedder_lr1(in_channels=256, out_channels=32)
        self.embeder_lr2 = Embedder_lr2(out_channels=40)

        self.predictorf1 = Predictor(num_in=40, classes=4)
        self.predictorf2 = Predictor(num_in=40, classes=4)
        self.predictorlr = Predictor(num_in=40, classes=1)

        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)
        self.bevpost = BevPost()

        self.use_quickcumsum = True

        # === Temporal fusion heads (캐시 기반) ===
        self.Z = int(self.nx[2].item())
        self.voxelC = self.camC * self.Z
        self.fuse_bev = TemporalConcat1x1(self.voxelC)  # voxelized 2D feature용
        self.fuse_txt = TemporalConcat1x1(256)  # SceneUnder 출력(256채널)용 (카메라 공용)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf["dbound"], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        _, C, imH, imW = x.shape
        B = self.bsize
        N = _ // B
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH, imW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)

        return x

    @staticmethod
    def visualize_bev_feat(bev: torch.Tensor, isBreak: bool = False, mode: str = "entropy"):
        import numpy as np, matplotlib.pyplot as plt, time, os

        if not os.path.exists("bev_prediction_images"):
            os.makedirs("bev_prediction_images")

        if mode == "entropy":
            bev_np = bev[0].permute(1, 2, 0).cpu().numpy()  # [200, 200, 4]
            exp_bev = np.exp(bev_np - np.max(bev_np, axis=2, keepdims=True))
            bev_softmax = exp_bev / np.sum(exp_bev, axis=2, keepdims=True)
            entropy = -np.sum(bev_softmax * np.log(bev_softmax + 1e-8), axis=2)
            plt.imsave(f"bev_prediction_images/bev_entropy_{time.time()}.png", entropy, cmap="viridis")
        elif mode == "integer_map":
            bev_np = bev[0].permute(1, 2, 0).cpu().numpy()  # [200, 200, 4]
            integer_map = np.argmax(bev_np, axis=2)
            plt.imsave(f"bev_prediction_images/bev_integer_map_{time.time()}.png", integer_map, cmap="viridis")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if isBreak:
            raise Exception("Image has been saved.")

    def stage_forward(
        self,
        x_single,
        rots_single,
        trans_single,
        intrins_single,
        post_rots_single,
        post_trans_single,
        prev_cache: dict | None,
    ):
        """
        prev_cache: {
          'bev': Tensor | None,
          'y_f': Tensor | None, 'y_l1': Tensor | None, 'y_r1': Tensor | None,
          'y_l2': Tensor | None, 'y_r2': Tensor | None,
        }
        """
        if prev_cache is None:
            prev_cache = {"bev": None, "y_f": None, "y_l1": None, "y_r1": None, "y_l2": None, "y_r2": None}

        img_feats = self.encoder(x_single)

        # --- BEV 경로 ---
        vox = self.get_voxels(
            img_feats,
            rots_single,
            trans_single,
            intrins_single,
            post_rots_single,
            post_trans_single,
        )
        bev_fused = self.fuse_bev(vox, prev_cache["bev"])

        # --- TXT 경로 ---
        y_all = self.sceneunder(img_feats)  # [B*N, 256, 8, 22]
        Ncams = self.data_aug_conf["Ncams"]
        y_l_1 = y_all[0::Ncams]  # [B, 256, 8, 22]
        y_f = y_all[1::Ncams]
        y_r_1 = y_all[2::Ncams]
        y_l_2 = y_all[3::Ncams]
        y_r_2 = y_all[5::Ncams]

        y_f_fused = self.fuse_txt(y_f, prev_cache["y_f"])
        y_l1_fused = self.fuse_txt(y_l_1, prev_cache["y_l1"])
        y_r1_fused = self.fuse_txt(y_r_1, prev_cache["y_r1"])
        y_l2_fused = self.fuse_txt(y_l_2, prev_cache["y_l2"])
        y_r2_fused = self.fuse_txt(y_r_2, prev_cache["y_r2"])

        # 다음 step용 캐시 갱신
        new_cache = {
            "bev": bev_fused,
            "y_f": y_f_fused,
            "y_l1": y_l1_fused,
            "y_r1": y_r1_fused,
            "y_l2": y_l2_fused,
            "y_r2": y_r2_fused,
        }

        return bev_fused, new_cache

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        T = x.shape[1]
        cache = None

        # 시간 순회하며 캐시 융합
        for t in range(T):
            bev_fused, cache = self.stage_forward(
                x[:, t].contiguous(),
                rots[:, t].contiguous(),
                trans[:, t].contiguous(),
                intrins[:, t].contiguous(),
                post_rots[:, t].contiguous(),
                post_trans[:, t].contiguous(),
                cache,
            )

        # 최종(융합된) BEV → seg logits
        bev_logits = self.bevencode(cache["bev"])  # [B, outC(=4), 200, 200]

        # BEVPost (crop) 추출
        bev_post = bev_logits.detach()
        bev_post = bev_post[:, :, 60:140, 56:144]
        bev_post = self.bevpost(bev_post)

        # === TXT 분기(융합된 최종 feature + bev_post 결합) ===
        # Front
        y_f = cache["y_f"]
        y_f = self.embeder_f1(y_f)  # [B, 32, 8, 22]
        y_f = torch.cat([y_f, bev_post], dim=1)  # [B, 40, 8, 22]
        y_f = self.embeder_f2(y_f)  # [B, 40]
        desc_f = self.predictorf1(y_f)  # [B, 4]
        act_f = self.predictorf2(y_f)  # [B, 4]

        # LR (front/back)
        def lr_head(y_map):
            y = self.embeder_lr1(y_map)  # [B, 32, 8, 22]
            y = torch.cat([y, bev_post], dim=1)  # [B, 40, 8, 22]
            y = self.embeder_lr2(y)  # [B, 40]
            return self.predictorlr(y)  # [B, 1]

        desc_l1 = lr_head(cache["y_l1"])
        desc_r1 = lr_head(cache["y_r1"])
        desc_l2 = lr_head(cache["y_l2"])
        desc_r2 = lr_head(cache["y_r2"])
        desc = torch.cat([desc_f, desc_l1, desc_l2, desc_r1, desc_r2], dim=1)  # [B, 8]

        return bev_logits, act_f, desc


def compile_model_bevtxt(bsize, grid_conf, data_aug_conf, outC):
    return torch.compile(BEV_TXT(bsize, grid_conf, data_aug_conf, outC))
