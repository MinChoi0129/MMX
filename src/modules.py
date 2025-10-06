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


def _resize_pos_embed_custom(pos_embed: torch.Tensor, new_hw: tuple, num_prefix_tokens: int = 1):
    # pos_embed: [1, T, C],   T = P + gh*gw
    # 반환:     [1, P + H*W, C]
    P = num_prefix_tokens
    prefix = pos_embed[:, :P] if P > 0 else pos_embed[:, :0]
    grid = pos_embed[:, P:] if P > 0 else pos_embed

    # 원래 grid 크기 추정(정사각형 가정)
    ghw = grid.shape[1]
    gh = int(ghw**0.5)
    gw = ghw // gh

    grid = grid.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)  # [1, C, gh, gw]
    grid = F.interpolate(grid, size=new_hw, mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_hw[0] * new_hw[1], -1)  # [1, H*W, C]
    return torch.cat([prefix, grid], dim=1)


class EncoderViT(nn.Module):
    """
    Drop-in replacement for Encoder:
      Input : [B, N, 3, 128, 352]
      Output: [B*N, 512, 8, 22]
    """

    def __init__(self, vit_name: str = "vit_small_patch16_224"):
        super().__init__()
        # 1) timm ViT (pretrained, head 제거)
        self.trunk = timm.create_model(vit_name, pretrained=True, num_classes=0)

        # 2) 128x352 입력 허용
        if hasattr(self.trunk, "patch_embed"):
            pe = self.trunk.patch_embed
            if hasattr(pe, "strict_img_size"):
                pe.strict_img_size = False
            if hasattr(pe, "img_size"):
                pe.img_size = (128, 352)

        # /16 그리드 크기 & 임베딩 차원
        patch = self.trunk.patch_embed.patch_size[0]  # 16
        self.H16, self.W16 = 128 // patch, 352 // patch  # 8, 22
        self.embed_dim = self.trunk.num_features  # e.g., 768

        # 3) /16, /32 정렬 → Up으로 융합 (EfficientNet Encoder와 동일한 채널/해상도로 맞춤)
        self.proj16 = nn.Conv2d(self.embed_dim, 256, kernel_size=1, bias=False)  # /16
        self.proj32 = nn.Conv2d(256, 256, kernel_size=1, bias=False)  # /32
        self.up1 = Up(256 + 256, 512)  # /32 → /16 skip concat → 512ch @ 8x22

    def _forward_vit_16x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [BN, 3, 128, 352] → ViT 토큰/블록 → [BN, C(embed), 8, 22]
        """
        trunk = self.trunk
        BN = x.shape[0]

        # patch_embed: 보통 [BN, L, C] (L=8*22)
        z = trunk.patch_embed(x)
        tokens = z if z.dim() == 3 else z.flatten(2).transpose(1, 2)

        # CLS 토큰
        if getattr(trunk, "cls_token", None) is not None:
            cls = trunk.cls_token.expand(BN, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # pos_embed를 (8,22)로 보간 후 더함
        if getattr(trunk, "pos_embed", None) is not None:
            pe = _resize_pos_embed_custom(trunk.pos_embed, (self.H16, self.W16), num_prefix_tokens=1)
            tokens = tokens + pe

        if getattr(trunk, "pos_drop", None) is not None:
            tokens = trunk.pos_drop(tokens)

        # Transformer blocks
        for blk in trunk.blocks:
            tokens = blk(tokens)
        if getattr(trunk, "norm", None) is not None:
            tokens = trunk.norm(tokens)

        # CLS 제거 후 2D 맵 복원
        if getattr(trunk, "cls_token", None) is not None:
            tokens = tokens[:, 1:, :]
        C = tokens.shape[-1]
        feat16 = tokens.transpose(1, 2).reshape(BN, C, self.H16, self.W16)  # [BN, C, 8, 22]
        return feat16

    def get_eff_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        EfficientNet Encoder와 동일한 인터페이스/형상:
          x: [B, N, 3, 128, 352] → [B*N, 512, 8, 22]
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # [BN, 3, 128, 352]

        f16 = self._forward_vit_16x(x)  # [BN, embed_dim, 8, 22]
        f16 = self.proj16(f16)  # [BN, 256, 8, 22]
        f32 = F.avg_pool2d(f16, 2, 2)  # [BN, 256, 4, 11]
        f32 = self.proj32(f32)  # [BN, 256, 4, 11]

        fused = self.up1(f32, f16)  # [BN, 512, 8, 22]
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_eff_depth(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b4")
        self.up1 = Up(448 + 160, 512)
        # 320+112 for b0/b1; 352+120 for b2; 384+136 for b3; 448+160 for b4; 512+176 for b5; 576+200 for b6, 640+224 for b7

        # Concat fusion을 위한 conv 레이어
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),  # 512 + 512 = 1024
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def get_eff_depth(self, x, deep_feature):
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
        out = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])

        # Concat fusion 로직
        if deep_feature is not None:
            # t >= 1: 과거 피처와 현재 피처를 concat 후 conv로 융합
            # deep_feature는 이전 시점의 최종 출력 [B*N, 512, 8, 22]
            fused = torch.cat([out, deep_feature], dim=1)  # [B*N, 1024, 8, 22]
            fused = self.fusion_conv(fused)  # [B*N, 512, 8, 22]
            return out, fused  # 융합된 결과를 반환하고 다음 시점을 위해 저장
        else:
            # t=0: 융합 없이 현재 피처만 사용
            return out, out  # 현재 피처를 반환하고 다음 시점을 위해 저장

    def forward(self, x, deep_feature):
        x, deep_feature = self.get_eff_depth(x, deep_feature)  # [B*D, 512, H=8, W=22]
        return x, deep_feature


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
