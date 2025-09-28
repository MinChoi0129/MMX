import torch
from torch import nn
from torch.nn import functional as F

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b4")
        self.up1 = Up(448 + 160, 512)
        # 320+112 for b0/b1; 352+120 for b2; 384+136 for b3; 448+160 for b4; 512+176 for b5; 576+200 for b6, 640+224 for b7

    def get_eff_depth(self, x):

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
        # print(x.shape) torch.Size([40, 512, 8, 22])

        return x

    def forward(self, x):
        x = self.get_eff_depth(x)
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
