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


def conv_nd(dim, *args, **kwargs):
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dim == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported convolution dim: {dim}")


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        if self.with_conv:
            self.conv = conv_nd(dim, in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

    def forward(self, x):
        scale = (2,) * self.dim
        x = torch.nn.functional.interpolate(x, scale_factor=scale, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        if self.with_conv:
            self.conv = conv_nd(dim, in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            if self.dim == 3:
                x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            else:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
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

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_nd(dim, in_channels,
                             out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = conv_nd(dim, out_channels,
                             out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(dim, in_channels,
                                             out_channels,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1)
            else:
                self.nin_shortcut = conv_nd(dim, in_channels,
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
            temb_proj = self.temb_proj(nonlinearity(temb))
            for _ in range(self.dim):
                temb_proj = temb_proj.unsqueeze(-1)
            h = h + temb_proj

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

        self.norm = Normalize(in_channels)
        self.q = conv_nd(dim, in_channels,
                         in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.k = conv_nd(dim, in_channels,
                         in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.v = conv_nd(dim, in_channels,
                         in_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)
        self.proj_out = conv_nd(dim, in_channels,
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

        # compute attention
        b, c, *spatial = q.shape
        q = q.reshape(b, c, -1)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b, c, -1) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, -1)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, *spatial)

        h_ = self.proj_out(h_)

        return x+h_