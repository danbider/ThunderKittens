import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange

B = 1
multiplier = 6
dim = 3072
num_heads = 24
img_in_dim, txt_in_dim = 4080, 512
head_dim = dim // num_heads
print(f"{head_dim=}")

TESTNAME = sys.argv[1]

# Modules and Weights
lin1 = nn.Linear(dim, 6 * dim, bias=True).cuda()
norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda() 
img_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda()
img_proj = nn.Linear(dim, dim).cuda()
q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
k_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()

if TESTNAME == 'ones':
    img = torch.randn(B, img_in_dim, dim, dtype=torch.bfloat16, device='cuda')
    txt = torch.randn(B, txt_in_dim, dim, dtype=torch.bfloat16, device='cuda')
    vec = torch.randn(B, dim, dtype=torch.float, device='cuda')
    pe  = torch.randn(B, 1, (img_in_dim + txt_in_dim), 64, 2, 2, dtype=torch.bfloat16, device='cuda')

else:
    print('Invalid test name')
    sys.exit(0)

def get_output(img, txt, vec, pe):
    # norm 
    img_modulated_init = norm1(img)

    # gemm
    silu_vec = nn.functional.silu(vec)
    out_img = lin1(silu_vec)[:, None, :].chunk(multiplier, dim=-1) 
    img_mod1 = out_img[:3]  
    img_modulated = (1 + img_mod1[1]) * img_modulated_init + img_mod1[0] 
    return img_modulated, img_mod1[0][0], img_mod1[1][0]

img_modulated_init, img_mod1_shift, img_mod1_scale = get_output(img, txt, vec, pe)


with open(f'{TESTNAME}.txt', 'w') as f:

    # inputs
    img_f = img.to(torch.float32).flatten().detach().cpu().numpy()
    txt_f = txt.to(torch.float32).flatten().detach().cpu().numpy()

    # intermediate outputs
    img_modulated_init_f = img_modulated_init.to(torch.float32).flatten().detach().cpu().numpy()

    # parameters 
    img_mod1_shift_f = img_mod1_shift.to(torch.float32).flatten().detach().cpu().numpy()
    img_mod1_scale_f = img_mod1_scale.to(torch.float32).flatten().detach().cpu().numpy()

    # save inputs 
    for i in trange(B*img_in_dim*dim):
        f.write(repr(img_f[i]))
        f.write(' ')

    # intermediate outputs
    for i in trange(B*img_in_dim*dim):
        f.write(repr(img_modulated_init_f[i]))
        f.write(' ')

    # save parameters 
    for i in trange(B*dim):
        f.write(repr(img_mod1_shift_f[i]))
        f.write(' ')
    for i in trange(B*dim):
        f.write(repr(img_mod1_scale_f[i]))
        f.write(' ')
