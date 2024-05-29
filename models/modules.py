import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange, repeat
from torch.nn.functional import mse_loss
from torch import nn, einsum
from .swin_module import *
from einops import rearrange
from einops import rearrange
import torch.nn.functional as F


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm  # B, N(H_sp*W_sp), C


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)  # B, H, W, C
    return img


def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition B', H_sp, W_sp, C
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp, W_sp, C)
    return img_perm  # B', H_sp, W_sp, C


def pixel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_channels = channels // (upscale_factor ** 2)
    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)
    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x


#   Channel Attention Block
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttentionBlock, self).__init__()
        self.reduction = reduction
        self.dct_layer = nn.AdaptiveAvgPool2d(1)  # DCTLayer(channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        y = self.dct_layer(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


#   Spatial Attention Block
class SpatialAttentionBlock(nn.Module):
    def __init__(self, channel):
        super(SpatialAttentionBlock, self).__init__()
        # Maximum pooling
        self.featureMap_max = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=0)
        )
        # Average pooling
        self.featureMap_avg = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.AvgPool2d(kernel_size=(5, 5), stride=(1, 1), padding=0)
        )

        # Deviation pooling
        # var = \sqrt(featureMap - featureMap_avg)^2

        # Dimensionality Reduction
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=channel * 4, out_channels=channel, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_max = self.featureMap_max(x)
        x_avg = self.featureMap_avg(x)
        x_var = torch.sqrt(torch.pow(x - x_avg, 2) + 1e-7)

        y = torch.cat([x_max, x_avg, x_var, x], dim=1)
        z = self.reduce_dim(y)
        return x * z

class SubstituteModule(nn.Module):
    def __init__(self, pan_original_channels=1, ms_original_channels=4, out_channels=16):
        super().__init__()
        self.combined_module = nn.Sequential(
            conv3x3(in_channels=ms_original_channels + pan_original_channels, out_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, combined_figure):
        """
        input:
            ms [B, ms_original_channels, H, W]
            pan [B, pan_original_channels, H, W]
        return:
            combined_figure:   [B, ms_original_channels + pan_original_channels, H, W]
        """

        combined_figure = self.combined_module(combined_figure)
        return combined_figure


# using the SwinModule to extract features

class SSModule(nn.Module):
    def __init__(self, n_feats=64, n_heads=4, head_dim=16, win_size=4, downsample_factor=1):
        super().__init__()

        # self-swin-attention module
        # after attention, there is a 2 downsample compared with input
        module = [
            SwinModule(in_channels=n_feats // downsample_factor, hidden_dimension=n_feats, layers=2,
                       downsample_factor=downsample_factor, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
            SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                       downsample_factor=1, num_heads=n_heads, head_dim=head_dim,
                       window_size=win_size, relative_pos_embedding=True, cross_attn=False),
        ]

        self.module = nn.Sequential(*module)

    def forward(self, x):
        """
            input_shape: (B, C, H, W)
            output_shape:   (B, n_feats, H//2, W//2)
        """
        feat = self.module(x)
        return feat  # B, n_feats, H, W
    


class IntermediateGuidanceModule(nn.Module):
    def __init__(self, n_feats=64, n_heads=4, head_dim=16, win_size=4, n_blocks=3):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = conv1x1(n_feats * 2, n_feats)

        self.ms_guidance_module = nn.ModuleList()
        self.pan_guidance_module = nn.ModuleList()
        for _ in range(n_blocks):
            self.ms_guidance_module.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                   downsample_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                   window_size=win_size, relative_pos_embedding=True, cross_attn=True))
            self.pan_guidance_module.append(SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                   downsample_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                   window_size=win_size, relative_pos_embedding=True, cross_attn=True))

        # self.feat_encoder = nn.Sequential(*feat_encoder)

    def forward(self, feat_ms, feat_pan, fusion):
        # fusion: fusion feature from last layer
        # feat: feat from ms or pan which can guide
        # fusion and feat have the same shape
        fusion_guide_ms = fusion
        fusion_guide_pan = fusion

        for i in range(self.n_blocks):
            fusion_guide_ms = self.ms_guidance_module[i](fusion_guide_ms, feat_ms)
            fusion_guide_pan = self.pan_guidance_module[i](fusion_guide_pan, feat_pan)

        fusion_pan_ms = fusion_guide_ms + fusion_guide_pan
        return fusion_pan_ms

# encoder using intermediate guidance block
class Encoder(nn.Module):
    def __init__(self, n_feats=64, n_heads=4, head_dim=16, win_size=4, downsample_factor=1, guidance = True):
        super().__init__()
        # if down_sampling 
        self.inter_encoder = SSModule(n_feats, n_heads, head_dim, win_size, downsample_factor)
        self.guidance = guidance
        if guidance:
            self.pan_encoder = SSModule(n_feats, n_heads, head_dim, win_size, downsample_factor)
            self.ms_encoder = SSModule(n_feats, n_heads, head_dim, win_size, downsample_factor)


    def forward(self, intermediate, pan_f = None, ms_f = None):
        intermediate_f = self.inter_encoder(intermediate)
        if self.guidance:
            pan_f = self.pan_encoder(pan_f)
            ms_f = self.ms_encoder(ms_f)
            return intermediate_f, pan_f, ms_f
        else:
            return intermediate_f 

class Decoder(nn.Module):
    def __init__(self, n_feats=64, n_heads=4, head_dim=16, win_size=4, downsample_factor=1):
        super().__init__()
        self.SSModule = SSModule(n_feats, n_heads, head_dim, win_size, downsample_factor)
    def forward(self, x):
        return self.SSModule(x) #  + self.convModule(x)


class FusionModule(nn.Module):
    def __init__(self, in_channels=16, out_channels=4):
        super(FusionModule, self).__init__()
        self.fusion_model = nn.Sequential(
            conv3x3(in_channels, in_channels),  # residual connect
            nn.LeakyReLU(0.2, inplace=True), conv3x3(in_channels, in_channels),
            nn.LeakyReLU(0.2, inplace=True), conv3x3(in_channels, out_channels),
            nn.Tanh())

    def forward(self, feature):
        return self.fusion_model(feature)


if __name__ == '__main__':
    pass
