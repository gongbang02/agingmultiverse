import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope

import os


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, info) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2) #[8, 24, 512, 128] + [8, 24, 900, 128] -> [8, 24, 1412, 128]
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # Feature extraction mode: save K, V features to disk for computing aging directions
        if info['save_feature']:
            feature_id = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id_double']) + '_' + info['type'] + '_' + 'V'
            
            # Apply attention mask to focus on age-related features
            scale = 1 / math.sqrt(q.shape[-1])
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
            sim = sim.softmax(dim=-1)
            sim_mask = sim[:,:,6:7,:]  # select the token related to age
            sim_mask = sim_mask > 0.0001
            sim_mask = sim_mask.permute(0, 1, 3, 2)
            v_masked = v.masked_fill(~sim_mask, 0)
            k_masked = k.masked_fill(~sim_mask, 0)
            
            info['feature_buffer_V'][feature_id] = v_masked.detach().cpu().half()
            info['feature_buffer_K'][feature_id.replace('_V', '_K')] = k_masked.detach().cpu().half()

        
        # Guided editing mode: apply precomputed aging directions
        elif not info['save_feature'] and info['inject']:
            feature_name_v = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id_double']) + '_' + info['type'] + '_' + 'V'
            feature_name_k = feature_name_v.replace('_V', '_K')
            
            if info['inverse']:
                # During inversion, store the original K, V in memory buffer
                info['feature'][feature_name_v] = v.cpu()
                info['feature'][feature_name_k] = k.cpu()
            else:
                # --- Apply Edits ---
                v_inversion = info['feature'][feature_name_v].cuda() # Load V from buffer
                k_inversion = info['feature'][feature_name_k].cuda() # Load K from buffer

                edit_weight = (info['target_age'] - info['input_age']) / (info['age_high'] - info['age_low'])
                guidance_strength = 0.5  # Adjust this value to control identity-vs-edit balance

                # Load and apply precomputed aging directions
                if info['direction_path']:
                    v_ref_path = os.path.join(info['direction_path'], feature_name_v + '.pth')
                    k_ref_path = os.path.join(info['direction_path'], feature_name_k + '.pth')

                    if os.path.exists(v_ref_path) and os.path.exists(k_ref_path):
                        v_ref = torch.load(v_ref_path, weights_only=True).cuda().to(torch.bfloat16)
                        k_ref = torch.load(k_ref_path, weights_only=True).cuda().to(torch.bfloat16)

                    # --- Apply Simulated Aging Regularization ---
                    k_inversion = k_inversion + edit_weight * k_ref

                # --- Apply K modulation ---
                k_cross_attn = torch.matmul(k, k_inversion.transpose(-1, -2)) / math.sqrt(k.shape[-1])
                k_cross_attn = torch.softmax(k_cross_attn, dim=-1)
                k = k + guidance_strength * torch.matmul(k_cross_attn, k_inversion)

                if info['direction_path']:
                    # --- Apply Simulated Aging Regularization ---
                    v_inversion = v_inversion + edit_weight * v_ref

                # --- Apply V projection ---
                v_inversion[:, :, :512, :] = 0.0
                dot_product = (v_inversion * v).sum(dim=-1, keepdim=True)
                v_norm_sq = (v * v).sum(dim=-1, keepdim=True)
                alpha = dot_product / (v_norm_sq + 1e-10)
                alpha = alpha.repeat(1, 1, 1, v.shape[-1])
                alpha[:, :, :512, :] = 1.0
                v = alpha * v
                # --- End Edits ---

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, info) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Feature extraction mode: save K, V features to disk for computing aging directions
        if info['save_feature']:
            feature_id = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id_single']) + '_' + info['type'] + '_' + 'V'
            
            # Apply attention mask to focus on age-related features
            scale = 1 / math.sqrt(q.shape[-1])
            sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * scale
            sim = sim.softmax(dim=-1)
            sim_mask = sim[:,:,6:7,:]  # select the token related to age
            sim_mask = sim_mask > 0.0001
            sim_mask = sim_mask.permute(0, 1, 3, 2)
            v_masked = v.masked_fill(~sim_mask, 0)
            k_masked = k.masked_fill(~sim_mask, 0)
            
            info['feature_buffer_V'][feature_id] = v_masked.detach().cpu().half()
            info['feature_buffer_K'][feature_id.replace('_V', '_K')] = k_masked.detach().cpu().half()
        
        # Guided editing mode: apply precomputed aging directions
        elif not info['save_feature'] and info['inject']:
            feature_name_v = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id_single']) + '_' + info['type'] + '_' + 'V'
            feature_name_k = feature_name_v.replace('_V', '_K')
            
            if info['inverse']:
                # During inversion, store the original K, V in memory buffer
                info['feature'][feature_name_v] = v.cpu()
                info['feature'][feature_name_k] = k.cpu()
            else:
                # --- Apply Edits ---
                v_inversion = info['feature'][feature_name_v].cuda() # Load V from buffer
                k_inversion = info['feature'][feature_name_k].cuda() # Load K from buffer

                edit_weight = (info['target_age'] - info['input_age']) / (info['age_high'] - info['age_low'])
                guidance_strength = 0.5  # Adjust this value to control identity-vs-edit balance

                # Load and apply precomputed aging directions
                if info['direction_path']:
                    v_ref_path = os.path.join(info['direction_path'], feature_name_v + '.pth')
                    k_ref_path = os.path.join(info['direction_path'], feature_name_k + '.pth')

                    if os.path.exists(v_ref_path) and os.path.exists(k_ref_path):
                        v_ref = torch.load(v_ref_path, weights_only=True).cuda().to(torch.bfloat16)
                        k_ref = torch.load(k_ref_path, weights_only=True).cuda().to(torch.bfloat16)

                    # --- Apply Simulated Aging Regularization ---
                    k_inversion = k_inversion + edit_weight * k_ref
                
                # --- Apply K modulation ---
                k_cross_attn = torch.matmul(k, k_inversion.transpose(-1, -2)) / math.sqrt(k.shape[-1])
                k_cross_attn = torch.softmax(k_cross_attn, dim=-1)
                k = k + guidance_strength * torch.matmul(k_cross_attn, k_inversion)

                if info['direction_path']:
                    # --- Apply Simulated Aging Regularization ---
                    v_inversion = v_inversion + edit_weight * v_ref
                
                # --- Apply V projection ---
                v_inversion[:, :, :512, :] = 0.0
                dot_product = (v_inversion * v).sum(dim=-1, keepdim=True)
                v_norm_sq = (v * v).sum(dim=-1, keepdim=True)
                alpha = dot_product / (v_norm_sq + 1e-10)
                alpha = alpha.repeat(1, 1, 1, v.shape[-1])
                alpha[:, :, :512, :] = 1.0
                v = alpha * v
                # --- End Edits ---

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output, info


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

