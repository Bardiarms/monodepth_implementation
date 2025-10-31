from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch: int, out_ch: int, kernel: int=3,
               stride: int=1, padding: int=1,
               activation: str= "relu") -> nn.Sequential:
    
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
   

# To produce single channel map
class DispHead(nn.Module):
    
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        return self.conv(x)
    
# Main model class implementing encoder-decoder.    
class DepthNet(nn.Module):
    
    def __init__(self, num_chs: List[int] = None, min_disp: float = 0.01):
        
        """
        num_chs: list of encoder channel widths. If None, uses a small default.
        min_disp: minimum disparity in pixels (avoids zero).
        """
        
        super().__init__()
        if num_chs is None:
            # Small channel width
            num_chs = [32, 64, 128, 256, 512]
            
        self.min_disp = float(min_disp)
                        
        
        # Encoder: convs with stride=2 downsampling
        self.enc1 = conv_block(3, num_chs[0], kernel=7, stride=2, padding=3)   # /2
        self.enc2 = conv_block(num_chs[0], num_chs[1], kernel=5, stride=2, padding=2)  # /4
        self.enc3 = conv_block(num_chs[1], num_chs[2], kernel=3, stride=2, padding=1)  # /8
        self.enc4 = conv_block(num_chs[2], num_chs[3], kernel=3, stride=2, padding=1)  # /16
        self.enc5 = conv_block(num_chs[3], num_chs[4], kernel=3, stride=2, padding=1)  # /32
        
        # Extra conv at the deepest level without modifying spatial size
        self.bottleneck = conv_block(num_chs[4], num_chs[4], kernel=2, stride=1, padding=1)
        
        # Decoder: upconv + concat skip -> conv
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
        
        # disparity heads at multiple scales
        self.disp_head_3 = DispHead(num_chs[2])  # 1/8  -> i4 has num_chs[2]
        self.disp_head_2 = DispHead(num_chs[1])  # 1/4  -> i3 has num_chs[1]
        self.disp_head_1 = DispHead(num_chs[0])  # 1/2  -> i2 has num_chs[0]
        self.disp_head_0 = DispHead(num_chs[0])  # full -> i1 has num_chs[0]
        
        self._init_weights()
        
        
    # Iterates through all submodules in the model
    # Better default initialization to speed convergence.
    def _init_weights(self) -> None:
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    
    # Convert raw_disp logits to positive disparity in *pixels*.
    def _scale_disp(self, raw_disp: torch.Tensor, width: int) -> torch.Tensor:
        
        sig = torch.sigmoid(raw_disp)
        max_disp = 0.3 * float(width) # Maximum disparity is set 30% of image width heuristically.
        disp = self.min_disp + sig * (max_disp - self.min_disp)
        
        return disp
    
    # Forward pass
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
        cat5 = torch.cat([u5, e4], dim=1)
        i5 = self.iconv5(cat5)
        
        u4 = self.up4(i5)
        u4 = F.interpolate(u4, size=(e3.shape[2], e3.shape[3]), mode='nearest')
        cat4 = torch.cat([u4, e3], dim=1)
        i4 = self.iconv4(cat4)
        
        u3 = self.up3(i4)
        u3 = F.interpolate(u3, size=(e2.shape[2], e2.shape[3]), mode='nearest')
        cat3 = torch.cat([u3, e2], dim=1)
        i3 = self.iconv3(cat3)
        
        u2 = self.up2(i3)
        u2 = F.interpolate(u2, size=(e1.shape[2], e1.shape[3]), mode='nearest')
        cat2 = torch.cat([u2, e1], dim=1)
        i2 = self.iconv2(cat2)
        
        u1 = self.up1(i2)
        i1 = self.iconv1(u1)
        
        # Disparity heads
        raw3 = self.disp_head_3(i4)
        raw2 = self.disp_head_2(i3)
        raw1 = self.disp_head_1(i2)
        raw0 = self.disp_head_0(i1)
        
        # scale to pixel disparity using input width
        disp3 = self._scale_disp(raw3, width=input_w)
        disp2 = self._scale_disp(raw2, width=input_w)
        disp1 = self._scale_disp(raw1, width=input_w)
        disp0 = self._scale_disp(raw0, width=input_w)
        
        outputs = {
            'disp_3': disp3,
            'disp_2': disp2,
            'disp_1': disp1,
            'disp_0': disp0
            }
        
        return outputs    