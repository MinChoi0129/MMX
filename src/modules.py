import torch
from torch import nn
from torch.nn import functional as F
import timm

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
from src.pretty_print import shprint


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


def _resize_pos_embed_custom(pos_embed: torch.Tensor, new_hw: tuple, num_prefix_tokens: int = 1) -> torch.Tensor:
    """
    pos_embed: [1, T, C] (T = num_prefix_tokens + gh*gw)
    new_hw: (H_new, W_new)
    num_prefix_tokens: 보통 1 (CLS)
    반환: [1, 1 + H_new*W_new, C]
    """
    assert pos_embed.dim() == 3 and pos_embed.size(0) == 1, "pos_embed shape must be [1, T, C]"
    T, C = pos_embed.shape[1], pos_embed.shape[2]
    H_new, W_new = new_hw

    if num_prefix_tokens > 0:
        prefix = pos_embed[:, :num_prefix_tokens]  # [1, P, C]
        grid = pos_embed[:, num_prefix_tokens:]  # [1, T-P, C]
    else:
        prefix = pos_embed.new_zeros(1, 0, C)
        grid = pos_embed

    # 기존 grid 크기 추정
    ghw = grid.shape[1]
    gh = int((ghw) ** 0.5)
    gw = ghw // gh
    assert gh * gw == ghw, "pos_embed grid tokens is not square"

    # [1, gh*gw, C] -> [1, C, gh, gw] -> 보간 -> [1, H_new*W_new, C]
    grid = grid.reshape(1, gh, gw, C).permute(0, 3, 1, 2)  # [1, C, gh, gw]
    grid = F.interpolate(grid, size=(H_new, W_new), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, H_new * W_new, C)  # [1, H_new*W_new, C]

    return torch.cat([prefix, grid], dim=1)


class EncoderViT(nn.Module):
    """
    Input : [B, N=6, 3, 128, 352]
    Output: [B*D, 512, 8, 22]
    """

    def __init__(self, D: int = 6, vit_name: str = "vit_base_patch16_224"):
        super().__init__()
        self.D = D
        self.trunk = timm.create_model(vit_name, pretrained=True, num_classes=0)

        # >>> 추가: 입력 해상도 강제 체크 해제 + 원하는 해상도로 설정 <<<
        if hasattr(self.trunk, "patch_embed"):
            pe = self.trunk.patch_embed
            if hasattr(pe, "strict_img_size"):
                pe.strict_img_size = False  # [핵심] 224 강제검사 비활성화
            if hasattr(pe, "img_size"):
                pe.img_size = (128, 352)  # 현재 파이프라인 입력 크기

        self.embed_dim = self.trunk.num_features  # e.g., 768
        patch = self.trunk.patch_embed.patch_size[0]  # 16
        assert patch == 16

        self.grid_h, self.grid_w = 128 // patch, 352 // patch  # 8, 22
        self.proj16 = nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False)
        self.proj32 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.up1 = Up(256 + 256, 512)
        self.depth_head = nn.Conv2d(512, 512 * D, kernel_size=1, bias=True)

    def _forward_vit_16x(self, x: torch.Tensor) -> torch.Tensor:
        """
        timm ViT에서 /16 특징맵 추출
        - patch_embed 가 3D([BN, L, C])나 4D([BN, C, H, W]) 모두 대응
        - pos_embed 를 (8,22)로 보간해서 더함
        반환: [BN, C, 8, 22]
        """
        trunk = self.trunk
        BN = x.shape[0]

        # 1) patch embedding
        z = trunk.patch_embed(x)
        if z.dim() == 3:
            # timm 기본: [BN, L, C]  (L = 8*22 = 176)
            BN_, L, C = z.shape
            assert BN_ == BN
            H16, W16 = self.grid_h, self.grid_w  # (8, 22)
            # tokens 그대로 사용
            tokens = z
        else:
            # 드물게 4D로 나오는 구현 대비
            BN_, C, H16, W16 = z.shape
            assert BN_ == BN and (H16, W16) == (self.grid_h, self.grid_w)
            tokens = z.flatten(2).transpose(1, 2)  # [BN, HW, C]

        # 2) CLS 토큰 붙이기
        if hasattr(trunk, "cls_token") and trunk.cls_token is not None:
            cls_tok = trunk.cls_token.expand(BN, -1, -1)  # [BN, 1, C]
            tokens = torch.cat((cls_tok, tokens), dim=1)  # [BN, 1+HW, C]

        # 3) pos_embed 보간 후 더하기
        if hasattr(trunk, "pos_embed") and trunk.pos_embed is not None:
            pos_embed = trunk.pos_embed  # [1, 1+H0*W0, C]
            pos_embed = _resize_pos_embed_custom(
                pos_embed, (self.grid_h, self.grid_w), num_prefix_tokens=1
            )  # [1, 1+HW, C] with HW=8*22
            tokens = tokens + pos_embed

        if hasattr(trunk, "pos_drop") and trunk.pos_drop is not None:
            tokens = trunk.pos_drop(tokens)

        # 4) transformer blocks
        for blk in trunk.blocks:
            tokens = blk(tokens)

        if hasattr(trunk, "norm") and trunk.norm is not None:
            tokens = trunk.norm(tokens)

        # 5) CLS 제거, 2D grid 복원
        if hasattr(trunk, "cls_token") and trunk.cls_token is not None:
            tokens = tokens[:, 1:, :]  # [BN, HW, C]

        C = tokens.shape[-1]
        feat = tokens.transpose(1, 2).reshape(BN, C, self.grid_h, self.grid_w)  # [BN, C, 8, 22]
        return feat

    def get_vit_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : [B, N, 3, 128, 352]
        Return: [B*D, 512, 8, 22]
        """
        B, N, C, H, W = x.shape
        assert (H, W) == (128, 352), "입력은 [*,*,3,128,352] 여야 합니다."
        x = x.view(B * N, C, H, W)  # [BN, 3, 128, 352]

        # /16 특징 (ViT)
        feat16 = self._forward_vit_16x(x)  # [BN, embed_dim, 8, 22]
        feat16 = self.proj16(feat16)  # [BN, 256, 8, 22]

        # /32 특징
        feat32 = F.avg_pool2d(feat16, kernel_size=2, stride=2)  # [BN, 256, 4, 11]
        feat32 = self.proj32(feat32)  # [BN, 256, 4, 11]

        # 업샘플 융합 → 512ch @ 8x22
        fused = self.up1(feat32, feat16)  # [BN, 512, 8, 22]

        # 카메라축 평균
        fused = fused.view(B, N, 512, 8, 22).mean(dim=1)  # [B, 512, 8, 22]

        # D 채널 생성 → [B*D, 512, 8, 22]
        out = self.depth_head(fused)  # [B, 512*D, 8, 22]
        out = out.view(B * self.D, 512, 8, 22)
        return out

    # Encoder 인터페이스 호환
    def get_eff_depth(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_vit_depth(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_vit_depth(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b4")
        self.up1 = Up(448 + 160, 512)
        # 320+112 for b0/b1; 352+120 for b2; 384+136 for b3; 448+160 for b4; 512+176 for b5; 576+200 for b6, 640+224 for b7

    def get_eff_depth(self, x):
        # Input: [B, N=6, C=3, H=128, W=352]
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        # return torch.Size([B*D, 512, 8, 22])
        return x

    def forward(self, x):
        x = self.get_eff_depth(x)  # [B*D, 512, H=8, W=22]
        return x


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):

        # Depth
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class BevPost(nn.Module):
    def __init__(self, in_channels=4, out_channels=8):
        super(BevPost, self).__init__()
        self.post = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 4), padding=0),
        )

    def forward(self, x):
        x = self.post(x)
        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class SceneUnder(nn.Sequential):
    def __init__(self, in_channels=512) -> None:
        super(SceneUnder, self).__init__(ASPP(in_channels, [12, 24, 36]))


class Embedder(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Embedder, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(out_channels * 22 * 8, out_channels, bias=True),  # for mobilenet in image of (320*160)
        )


class Embedder_lr1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Embedder_lr1, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Embedder_lr2(nn.Sequential):
    def __init__(self, out_channels):
        super(Embedder_lr2, self).__init__(
            nn.Flatten(),
            nn.Linear(out_channels * 22 * 8, out_channels, bias=True),  # for mobilenet in image of (320*160)
        )


class Embedder_f1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Embedder_f1, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()
        )


class Embedder_f2(nn.Sequential):
    def __init__(self, out_channels):
        super(Embedder_f2, self).__init__(nn.Flatten(), nn.Linear(out_channels * 22 * 8, out_channels, bias=True))


class Predictor(nn.Sequential):
    def __init__(self, num_in, classes):
        super(Predictor, self).__init__(nn.Linear(num_in, classes, bias=True))
