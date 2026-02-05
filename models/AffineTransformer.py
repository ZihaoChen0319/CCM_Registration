import math

import numpy as np
import torch
import torch.nn as nn
from timm.layers import Mlp, DropPath, trunc_normal_

from .Module import FeatureExtractor, para_to_matrix_affine_2d, para_to_matrix_affine_3d


#################################################################################
#                             Transformer Blocks                                #
#################################################################################
class SelfAttenTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, drop_path=0., mlp_ratio=4.0, conditional=True):
        super(SelfAttenTransBlock, self).__init__()
        self.conditional = conditional

        self.norm_1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_1 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        if self.conditional:
            self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attn_2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm_3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, c):
        x_norm = self.norm_1(x)
        x = x + self.drop_path(self.attn_1(x_norm, x_norm, x_norm, need_weights=False)[0])

        if self.conditional:
            x_norm = self.norm_2(x)
            x = x + self.drop_path(self.attn_2(x_norm, c, c, need_weights=False)[0])

        x = x + self.drop_path(self.mlp(self.norm_3(x)))
        return x


class CrossAttenTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, drop_path=0., mlp_ratio=4.0, conditional=True):
        super(CrossAttenTransBlock, self).__init__()
        self.conditional = conditional

        self.norm_1_1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_1_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_1 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        if self.conditional:
            self.norm_2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.attn_2 = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        self.norm_3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x1, x2, c):
        x1_norm, x2_norm = self.norm_1_1(x1), self.norm_1_2(x2)
        x = x1 + self.drop_path(self.attn_1(x1_norm, x2_norm, x2_norm, need_weights=False)[0])

        if self.conditional:
            x_norm = self.norm_2(x)
            x = x + self.drop_path(self.attn_2(x_norm, c, c, need_weights=False)[0])

        x = x + self.drop_path(self.mlp(self.norm_3(x)))
        return x


class SATRNetCornea(nn.Module):
    def __init__(self, image_size=(384, 384), num_channels_extractor=(1, 32, 64, 128, 256),
                 num_blocks_extractor=(2, 2, 2, 2), num_heads=4, depth_cross=4, depth_self=4,
                 norm='BatchNorm'):
        super(SATRNetCornea, self).__init__()
        """
        Step-Aware Transformer Registration Network
        """

        # image patch embedding
        self.feature_extractor = FeatureExtractor(
            image_size=image_size,
            num_channels=num_channels_extractor,
            num_blocks=num_blocks_extractor,
            norm=norm
        )

        num_level = len(num_blocks_extractor)
        patch_grid_size = [image_size[i] // 2 ** num_level for i in range(len(image_size))]
        hidden_size = num_channels_extractor[-1]

        # step embedding
        self.step_embedder = TimestepEmbedder(hidden_size)

        # patch position embedding
        pos_embed = torch.from_numpy(get_2d_sin_cos_pos_embed(hidden_size, patch_grid_size))
        self.pos_embed = nn.Parameter(pos_embed.float().unsqueeze(0), requires_grad=False)  # (1, HW, C)

        # cross attention transformer block
        self.cross_attn_1 = nn.ModuleList([
            CrossAttenTransBlock(hidden_size, num_heads, conditional=True) for _ in range(depth_cross)
        ])
        self.cross_attn_2 = nn.ModuleList([
            CrossAttenTransBlock(hidden_size, num_heads, conditional=True) for _ in range(depth_cross)
        ])

        # self attention transformer block
        self.self_attn = nn.ModuleList([
            SelfAttenTransBlock(hidden_size, num_heads, conditional=True) for _ in range(depth_self)
        ])

        # regressor
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_size, 6),
            nn.Tanh()
        )

    def forward(self, image_moving, image_fixed, step):
        feat_moving = self.feature_extractor(image_moving)  # (B, C, H, W)
        feat_fixed = self.feature_extractor(image_fixed)

        B, C, H, W = feat_moving.shape
        feat_moving = feat_moving.view(B, C, H*W).permute(0, 2, 1) + self.pos_embed  # (B, HW, C)
        feat_fixed = feat_fixed.view(B, C, H*W).permute(0, 2, 1) + self.pos_embed

        step_embedding = self.step_embedder(step).unsqueeze(1)  # (B, 1, C)

        for block_1, block_2 in zip(self.cross_attn_1, self.cross_attn_2):
            x1 = block_1(feat_moving, feat_fixed, step_embedding)  # (B, HW, C)
            x2 = block_2(feat_fixed, feat_moving, step_embedding)
            feat_moving, feat_fixed = x1, x2

        x = x1 + x2  # (B, HW, C)
        for block in self.self_attn:
            x = block(x, step_embedding)

        x = self.head(x.mean(dim=1))

        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=100):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t = t.float()
        # noisy_t = t + torch.randn_like(t) * 0.1
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_3d_sin_cos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height, width and depth
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid_d = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    assert embed_dim % 3 == 0
    emb_h = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W, D/3)
    emb_w = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/3)
    emb_d = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 3, grid[2])
    pos_embed = np.concatenate([emb_h, emb_w, emb_d], axis=1)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sin_cos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    assert embed_dim % 2 == 0
    emb_h = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sin_cos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sin_cos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
