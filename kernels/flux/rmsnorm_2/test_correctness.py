
import torch 
from torch import Tensor, nn
from einops import rearrange
from dataclasses import dataclass
import time

import thunderkittens as tk

b, s, dim = 1, 4080, 3072
img_in_dim, txt_in_dim = 4080, 512
img = torch.randn(b, img_in_dim, dim, dtype=torch.float, device='cuda')
txt = torch.randn(b, txt_in_dim, dim, dtype=torch.float, device='cuda')
vec = torch.randn(b, dim, dtype=torch.float, device='cuda')
pe = torch.randn(b, 1, (img_in_dim + txt_in_dim), 64, 2, 2, dtype=torch.float, device='cuda')


def simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe):
    multiplier = 6
    dim = 3072
    num_heads = 24
    head_dim = dim // num_heads

    # Modules and Weights
    lin1 = nn.Linear(dim, 6 * dim, bias=True).cuda().to(torch.bfloat16)
    lin2 = nn.Linear(dim, 6 * dim, bias=True).cuda().to(torch.bfloat16)
    norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda() 
    norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    img_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda().to(torch.bfloat16)

    # Img prep
    out_img = lin1(nn.functional.silu(vec).to(torch.bfloat16))[:, None, :].chunk(multiplier, dim=-1)
    img_mod1, img_mod2 = out_img[:3], out_img[3:]  

    torch.cuda.synchronize()
    start = time.time()
    img_modulated = norm1(img.to(torch.bfloat16))
    img_modulated = (1 + img_mod1[1].to(torch.bfloat16)) * img_modulated.to(torch.bfloat16) + img_mod1[0].to(torch.bfloat16)
    torch.cuda.synchronize()
    end = time.time()
    print(f"torch (s) = {end-start}")

    print(f"Starting the layrnorm step.")
    img_modulated_tk = torch.empty_like(img_modulated)
    print(f"Created an empty tensor of shape: {img_modulated_tk.shape}")
    torch.cuda.synchronize()
    start = time.time()
    tk.fused_flux_layernorm(
        img.to(torch.bfloat16).contiguous(), 
        img_mod1[0][0][0].to(torch.bfloat16).contiguous(), 
        img_mod1[1][0][0].to(torch.bfloat16).contiguous(),
        img_modulated_tk.to(torch.bfloat16).contiguous()
    )
    torch.cuda.synchronize()
    end = time.time()
    print(f"tk (s) = {end-start}")
    diff = torch.norm(img_modulated - img_modulated_tk).max()
    print(f"Diff: {diff}; TODO: Convert to floats.")

    # RMS norm
    print(f"\nStarting the rmsnorm:")
    img_qkv = img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)

    torch.cuda.synchronize()
    start = time.time()
    mean = torch.mean(img_q**2, dim=-1, keepdim=True)
    rrms = torch.rsqrt(mean + 1e-6)
    img_q_ref = (img_q * rrms) * q_img_rms_norm_scale
    torch.cuda.synchronize()
    end = time.time()
    print(f"torch (s) = {end-start}")

    img_q_tk = torch.empty_like(img_q_ref).to(torch.bfloat16).contiguous()
    print(f"Created an empty tensor of shape: {img_q_tk.shape}")
    torch.cuda.synchronize()
    start = time.time()
    tk.fused_flux_rmsnorm(
        img_q.to(torch.bfloat16).contiguous(), 
        q_img_rms_norm_scale.to(torch.bfloat16).contiguous(), 
        img_q_tk
    )
    torch.cuda.synchronize()
    end = time.time()
    print(f"tk (s) = {end-start}")
    diff = torch.norm(img_q_ref - img_q_tk).max() 
    print(f"Diff: {diff=}")

    return img_q_ref

simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe)

