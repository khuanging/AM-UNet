import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from einops import rearrange, repeat
from functools import partial
from collections import OrderedDict
# from mmengine.model import BaseModule
# from mmengine.model.weight_init import kaiming_init, constant_init, trunc_normal_init
from timm.layers import trunc_normal_, DropPath, to_2tuple


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetUpsample(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetUpsample, self).__init__()
        self.up = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.up(input)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class CCSAttention2(nn.Module):

    def __init__(self, channel=256, reduction=16, G=4):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(channel, channel // 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                constant_init(m.weight, val=1.0)
                constant_init(m.bias, val=0)
            elif isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, val=0)

    def channel_shuffle(self, x, groups):
        b, c, d, h, w = x.shape
        x = x.reshape(b, groups, -1, d, h, w)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, d, h, w)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x

    def forward(self, x):
        b, c, d, h, w = x.size()
        # group into subfeatures
        x = x.contiguous().view(b * self.G, -1, d, h, w)  # bs*G,c//G,d,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),d,h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),d,h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),d,h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),d,h,w

        # concatenate along channel axis
        out = torch.cat([x_spatial, x_spatial], dim=1)  # bs*G,c//G,d,h,w
        out = out.reshape(b, -1, d, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        # out = self.channels_(out)
        return x_channel


class CCSAttention(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        self.proj = nn.Conv3d(channel, channel, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(channel)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(channel, channel, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(channel)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(channel, channel, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(channel)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(channel, channel, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(channel)
        self.nonliner4 = nn.ReLU()

        self.proj5 = nn.Conv3d(channel, channel // 2, 1, 1, 0)
        self.norm5 = nn.InstanceNorm3d(channel)
        self.nonliner5 = nn.ReLU()

    def forward(self, x):
        x_residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)

        x = x + x_residual

        x = self.proj5(x)
        x = self.norm5(x)
        x = self.nonliner5(x)

        return x


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        # self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)  #TODO,1e-6?
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv3d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ConvolutionalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention, self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=in_channels)

        self.kv1 = nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1, 1))
        self.kv2 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7, 1))
        self.kv3 = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 1, 7))
        trunc_normal_init(self.kv1, std=0.001)
        trunc_normal_init(self.kv2, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)

    # def _act_dn(self, x):
    #     x_shape = x.shape  # n,c_inter,h,w
    #     d, h, w = x_shape[2], x_shape[3], x_shape[4]
    #     x = x.reshape(
    #         [x_shape[0], self.num_heads, self.inter_channels // self.num_heads,
    #          -1])  # n,c_inter,h,w -> n,heads,c_inner//heads,hw
    #     x = F.softmax(x, dim=3)
    #     x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
    #     x = x.reshape([x_shape[0], self.inter_channels, d, h, w])
    #     return x

    def _act_dn(self, x):
        x_shape = x.shape  # (n, c_inter, d, h, w)
        b, c_inter, d, h, w = x_shape

        # 将输入张量重塑为适合多头注意力的形状
        x = x.reshape([b, self.num_heads, self.inter_channels // self.num_heads, d * h * w])

        # 在最后一个维度上进行 Softmax 操作
        x = F.softmax(x, dim=-1)

        # 归一化
        x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-06)

        # 恢复原始形状
        x = x.reshape([b, c_inter, d, h, w])

        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv3d(x, self.kv1, bias=None, stride=1, padding=(3, 0, 0))
        x1 = self._act_dn(x1)
        x1 = F.conv3d(x1, self.kv1.transpose(1, 0), bias=None, stride=1, padding=(3, 0, 0))
        x2 = F.conv3d(x, self.kv2, bias=None, stride=1, padding=(0, 3, 0))
        x2 = self._act_dn(x2)
        x2 = F.conv3d(x2, self.kv2.transpose(1, 0), bias=None, stride=1, padding=(0, 3, 0))
        x3 = F.conv3d(x, self.kv3, bias=None, stride=1, padding=(0, 0, 3))
        x3 = self._act_dn(x3)
        x3 = F.conv3d(x3, self.kv3.transpose(1, 0), bias=None, stride=1, padding=(0, 0, 3))
        x = x1 + x2 + x3

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class PatchEmbed(nn.Module):
    """ Volume to Patch Embedding
    """

    def __init__(self, vol_size: Tuple[int, int, int], patch_size: Tuple[int, int, int], in_chans: int,
                 embed_dim: int = 768):
        super().__init__()
        vol_size = tuple(vol_size)  # Ensure it's a tuple for safety
        patch_size = tuple(patch_size)
        num_patches = (vol_size[2] // patch_size[2]) * (vol_size[1] // patch_size[1]) * (vol_size[0] // patch_size[0])
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert D == self.vol_size[0] and H == self.vol_size[1] and W == self.vol_size[2], \
            f"Input volume size ({D}*{H}*{W}) doesn't match model ({self.vol_size[0]}*{self.vol_size[1]}*{self.vol_size[2]})."

        x = self.proj(x)
        B, C, D, H, W = x.shape

        x = x.permute(0, 2, 3, 4, 1)  # Move channels to the last dimension
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        return x


# class LGLBlocks(nn.Module):
#     def __init__(self, depth=3, img_size=(32, 128, 128), in_chans=3, embed_dim=384,
#                  head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, sr_ratios=2, **kwargs
#                  ):
#         super(LGLBlocks, self).__init__()
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other Models
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#
#         self.patch_embed = PatchEmbed(
#             vol_size=img_size, patch_size=[2, 2, 2], in_chans=in_chans, embed_dim=embed_dim)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         num_heads = embed_dim // head_dim
#         self.blocks = nn.ModuleList([
#             LGLBlock(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
#                 sr_ratio=sr_ratios) for i in range(depth)])
#
#         self.norm = nn.BatchNorm3d(embed_dim)
#
#         # Representation layer
#         if representation_size:
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(embed_dim, representation_size)),
#                 ('act', nn.Tanh())
#             ]))
#         else:
#             self.pre_logits = nn.Identity()
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#
#         return x

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.contiguous().view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class PatchExpand3D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpand3D, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, self.dim * (dim_scale * 2), bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        B, D, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale, c=C // self.dim_scale)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x

class SimAM3D(nn.Module):
    def __init__(self, channel=None, e_lambda=1e-4):
        super(SimAM3D, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.conv_proj = nn.Sequential(
            nn.Conv3d(channel, channel // 2, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm3d(channel // 2),
            nn.ReLU(),
        )

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s


    @staticmethod
    def get_module_name():
        return "simam3d"

    def forward(self, x):
        b, c, d, h, w = x.size()  # 3D输入的形状：(batch_size, channels, depth, height, width)
        x1 = self.conv_proj(x)
        n = d * h * w - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5
        y1 = self.conv_proj(self.activation(y))
        return x1 * y1



if __name__ == '__main__':
    from thop import profile
    x = torch.rand(4, 24, 32, 112, 112).cuda()
    model = SimAM3D(channels=24).cuda()
    out = model(x)
    input = torch.randn([4, 24, 24, 24, 24]).cuda()
    macs, params = profile(model.cuda(), inputs=(input,))
    print(macs, params)
    print(out.shape)

