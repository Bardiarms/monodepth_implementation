from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, kernel: int = 3,
               stride: int = 1, padding: int = 1,
               activation: str = "relu") -> nn.Sequential:

    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=True)]

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(alpha=1.0, inplace=True))
    else:
        raise ValueError("unsupported activation")

    return nn.Sequential(*layers)


def upconv(inc_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv_block(inc_ch, out_ch, kernel=3, stride=1, padding=1)
    )


class DispHead(nn.Module):
    """
    Outputs 2-channel disparity logits:
      channel 0 -> left disparity logits
      channel 1 -> right disparity logits
    """
    def __init__(self, in_ch: int, out_ch: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthNet(nn.Module):
    def __init__(self, num_chs: List[int] = None, min_disp: float = 0.01):
        super().__init__()
        if num_chs is None:
            num_chs = [32, 64, 128, 256, 512]

        self.min_disp = float(min_disp)

        # Encoder
        self.enc1 = conv_block(3, num_chs[0], kernel=7, stride=2, padding=3)     # /2
        self.enc2 = conv_block(num_chs[0], num_chs[1], kernel=5, stride=2, padding=2)  # /4
        self.enc3 = conv_block(num_chs[1], num_chs[2], kernel=3, stride=2, padding=1)  # /8
        self.enc4 = conv_block(num_chs[2], num_chs[3], kernel=3, stride=2, padding=1)  # /16
        self.enc5 = conv_block(num_chs[3], num_chs[4], kernel=3, stride=2, padding=1)  # /32

        self.bottleneck = conv_block(num_chs[4], num_chs[4], kernel=3, stride=1, padding=1)

        # Decoder
        self.up5 = upconv(num_chs[4], num_chs[3])
        self.iconv5 = conv_block(num_chs[3] + num_chs[3], num_chs[3], activation='elu')

        self.up4 = upconv(num_chs[3], num_chs[2])
        self.iconv4 = conv_block(num_chs[2] + num_chs[2], num_chs[2], activation='elu')

        self.up3 = upconv(num_chs[2], num_chs[1])
        self.iconv3 = conv_block(num_chs[1] + num_chs[1], num_chs[1], activation='elu')

        self.up2 = upconv(num_chs[1], num_chs[0])
        self.iconv2 = conv_block(num_chs[0] + num_chs[0], num_chs[0], activation='elu')

        self.up1 = upconv(num_chs[0], num_chs[0])
        self.iconv1 = conv_block(num_chs[0], num_chs[0], activation='elu')

        # 2-channel heads at 4 scales
        self.disp_head_3 = DispHead(num_chs[2], out_ch=2)  # from i4 (1/8)
        self.disp_head_2 = DispHead(num_chs[1], out_ch=2)  # from i3 (1/4)
        self.disp_head_1 = DispHead(num_chs[0], out_ch=2)  # from i2 (1/2)
        self.disp_head_0 = DispHead(num_chs[0], out_ch=2)  # from i1 (full)

        self._init_weights()

        for head in [self.disp_head_0, self.disp_head_1, self.disp_head_2, self.disp_head_3]:
            nn.init.constant_(head.conv.bias, -2.0)  # sigmoid(-2) ~ 0.12 → smaller initial disp


    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _scale_disp(self, raw_disp: torch.Tensor, width: int) -> torch.Tensor:
        """
        raw_disp: (B, 1, H, W) logits
        returns:  (B, 1, H, W) disparity in pixels
        """
        sig = torch.sigmoid(raw_disp)
        max_disp = 0.3 * float(width)
        disp = self.min_disp + sig * (max_disp - self.min_disp)
        return disp

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_h, input_w = x.shape[2], x.shape[3]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        # Decoder
        u5 = self.up5(b)
        u5 = F.interpolate(u5, size=(e4.shape[2], e4.shape[3]), mode='nearest')
        i5 = self.iconv5(torch.cat([u5, e4], dim=1))

        u4 = self.up4(i5)
        u4 = F.interpolate(u4, size=(e3.shape[2], e3.shape[3]), mode='nearest')
        i4 = self.iconv4(torch.cat([u4, e3], dim=1))

        u3 = self.up3(i4)
        u3 = F.interpolate(u3, size=(e2.shape[2], e2.shape[3]), mode='nearest')
        i3 = self.iconv3(torch.cat([u3, e2], dim=1))

        u2 = self.up2(i3)
        u2 = F.interpolate(u2, size=(e1.shape[2], e1.shape[3]), mode='nearest')
        i2 = self.iconv2(torch.cat([u2, e1], dim=1))

        u1 = self.up1(i2)
        i1 = self.iconv1(u1)

        # Raw 2-channel disparity logits at each scale
        raw3 = self.disp_head_3(i4)  # (B,2,h,w)
        raw2 = self.disp_head_2(i3)
        raw1 = self.disp_head_1(i2)
        raw0 = self.disp_head_0(i1)

        # Split + scale (each becomes (B,1,h,w))
        def split_and_scale(raw2ch: torch.Tensor):
            raw_l = raw2ch[:, 0:1, :, :]
            raw_r = raw2ch[:, 1:2, :, :]
            disp_l = self._scale_disp(raw_l, width=input_w)
            disp_r = self._scale_disp(raw_r, width=input_w)
            return disp_l, disp_r

        disp_l_3, disp_r_3 = split_and_scale(raw3)
        disp_l_2, disp_r_2 = split_and_scale(raw2)
        disp_l_1, disp_r_1 = split_and_scale(raw1)
        disp_l_0, disp_r_0 = split_and_scale(raw0)

        return {
            "disp_l_3": disp_l_3, "disp_r_3": disp_r_3,
            "disp_l_2": disp_l_2, "disp_r_2": disp_r_2,
            "disp_l_1": disp_l_1, "disp_r_1": disp_r_1,
            "disp_l_0": disp_l_0, "disp_r_0": disp_r_0,
        }