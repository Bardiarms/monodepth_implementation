from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Conv3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = Conv3x3(in_ch, out_ch)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DispHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResnetEncoder(nn.Module):
    """
    Returns features at multiple scales:
      f0: 1/2  (after maxpool)
      f1: 1/4  (layer1)
      f2: 1/8  (layer2)
      f3: 1/16 (layer3)
      f4: 1/32 (layer4)
    """
    def __init__(self, num_layers: int = 18, pretrained: bool = True):
        super().__init__()
        if num_layers == 18:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.num_ch = [64, 64, 128, 256, 512]
        elif num_layers == 50:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.num_ch = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError("num_layers must be 18 or 50")

        # take layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))  # /2
        f0 = self.maxpool(x)                    # /4-ish (depending on conv1 stride; in resnet it's /4)
        f1 = self.layer1(f0)                    # /4
        f2 = self.layer2(f1)                    # /8
        f3 = self.layer3(f2)                    # /16
        f4 = self.layer4(f3)                    # /32
        return [f0, f1, f2, f3, f4]


class DepthDecoder(nn.Module):
    """
    Monodepth2-style decoder producing disp at 4 scales:
      disp_3 at 1/8
      disp_2 at 1/4
      disp_1 at 1/2
      disp_0 at full
    """
    def __init__(self, enc_ch: List[int], min_disp: float = 0.01):
        super().__init__()
        self.min_disp = float(min_disp)

        # decoder channels (common choice)
        dec_ch = [16, 32, 64, 128, 256]

        # upconvs: from deepest to shallow
        self.up4 = UpConv(enc_ch[4], dec_ch[4])                 # /16
        self.iconv4 = ConvBlock(dec_ch[4] + enc_ch[3], dec_ch[4])

        self.up3 = UpConv(dec_ch[4], dec_ch[3])                 # /8
        self.iconv3 = ConvBlock(dec_ch[3] + enc_ch[2], dec_ch[3])

        self.up2 = UpConv(dec_ch[3], dec_ch[2])                 # /4
        self.iconv2 = ConvBlock(dec_ch[2] + enc_ch[1], dec_ch[2])

        self.up1 = UpConv(dec_ch[2], dec_ch[1])                 # /2
        self.iconv1 = ConvBlock(dec_ch[1] + enc_ch[0], dec_ch[1])

        self.up0 = UpConv(dec_ch[1], dec_ch[0])                 # /1
        self.iconv0 = ConvBlock(dec_ch[0], dec_ch[0])

        # disparity heads at scales
        self.disp3 = DispHead(dec_ch[3])  # /8  (after iconv3)
        self.disp2 = DispHead(dec_ch[2])  # /4  (after iconv2)
        self.disp1 = DispHead(dec_ch[1])  # /2  (after iconv1)
        self.disp0 = DispHead(dec_ch[0])  # /1  (after iconv0)

        # init disparity head biases to start small
        for head in [self.disp0, self.disp1, self.disp2, self.disp3]:
            nn.init.constant_(head.conv.bias, -2.0)

    def _scale_disp(self, raw_disp: torch.Tensor, width: int) -> torch.Tensor:
        sig = torch.sigmoid(raw_disp)
        max_disp = 0.3 * float(width)
        return self.min_disp + sig * (max_disp - self.min_disp)

    def forward(self, feats: List[torch.Tensor], input_w: int) -> Dict[str, torch.Tensor]:
        f0, f1, f2, f3, f4 = feats

        x = self.up4(f4)
        x = F.interpolate(x, size=(f3.shape[2], f3.shape[3]), mode="nearest")
        x = self.iconv4(torch.cat([x, f3], dim=1))

        x = self.up3(x)
        x = F.interpolate(x, size=(f2.shape[2], f2.shape[3]), mode="nearest")
        x = self.iconv3(torch.cat([x, f2], dim=1))
        disp_3 = self._scale_disp(self.disp3(x), width=input_w)

        x = self.up2(x)
        x = F.interpolate(x, size=(f1.shape[2], f1.shape[3]), mode="nearest")
        x = self.iconv2(torch.cat([x, f1], dim=1))
        disp_2 = self._scale_disp(self.disp2(x), width=input_w)

        x = self.up1(x)
        x = F.interpolate(x, size=(f0.shape[2], f0.shape[3]), mode="nearest")
        x = self.iconv1(torch.cat([x, f0], dim=1))
        disp_1 = self._scale_disp(self.disp1(x), width=input_w)

        x = self.up0(x)
        x = self.iconv0(x)
        disp_0 = self._scale_disp(self.disp0(x), width=input_w)

        return {"disp_3": disp_3, "disp_2": disp_2, "disp_1": disp_1, "disp_0": disp_0}


class DepthNet(nn.Module):
    """
    Stereo-only Monodepth2-style network:
      left image -> disp_0..disp_3
    """
    def __init__(self, resnet_layers: int = 18, pretrained: bool = True, min_disp: float = 0.01):
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=resnet_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(enc_ch=self.encoder.num_ch, min_disp=min_disp)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_w = x.shape[3]
        feats = self.encoder(x)
        return self.decoder(feats, input_w=input_w)