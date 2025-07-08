"""borrowed and modified from https://github.com/CompVis/taming-transformers"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import math
import numpy as np


def nonlinearity(x):
    # swish
    return F.silu(x, inplace=True)  # x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        if self.with_conv:
            self.conv = Conv(in_channels,
                             in_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = Conv(in_channels,
                             in_channels,
                             kernel_size=3,
                             stride=2,
                             padding=0)

    def forward(self, x):
        if self.with_conv:
            spatial = x.shape[-self.dim:]
            pads = [1] * self.dim
            for i, s in enumerate(spatial[::-1]):
                if s + pads[i] < 3:
                    pads[i] = 3 - s
            pad = []
            for p in pads:
                pad.extend([0, p])
            x = F.pad(x, pad, mode="constant", value=0)
            if all(s >= 1 for s in spatial):
                x = self.conv(x)
        else:
            pool = F.avg_pool3d if self.dim == 3 else F.avg_pool2d
            if all(s >= 2 for s in x.shape[-self.dim:]):
                x = pool(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, dim=2):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.checkpointing = False
        self.dim = dim
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv(in_channels,
                          out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = Conv(out_channels,
                          out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv(in_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
            else:
                self.nin_shortcut = Conv(in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

    def _forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            add_shape = (h.shape[0], -1) + (1,)*self.dim
            h = h + self.temb_proj(nonlinearity(temb)).view(*add_shape)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

    def forward(self, x, temb):
        if self.checkpointing and self.training:
            out = checkpoint(self._forward, x, temb)
        else:
            out = self._forward(x, temb)
        return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d

        self.norm = Normalize(in_channels)
        self.q = Conv(in_channels,
                      in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        self.k = Conv(in_channels,
                      in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        self.v = Conv(in_channels,
                      in_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0)
        self.proj_out = Conv(in_channels,
                             in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        spatial = q.shape[2:]
        b, c = q.shape[:2]
        hw = int(np.prod(spatial))
        q = q.reshape(b, c, hw).permute(0, 2, 1)
        k = k.reshape(b, c, hw)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, hw)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, *spatial)

        h_ = self.proj_out(h_)

        return x+h_