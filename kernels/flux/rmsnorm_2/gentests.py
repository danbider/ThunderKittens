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
    rms_input_img_q = torch.randn(B, num_heads, img_in_dim, head_dim, dtype=torch.bfloat16, device='cuda')/head_dim

else:
    print('Invalid test name')
    sys.exit(0)

def get_output(rms_input_img_q):
    rrms = torch.rsqrt(torch.mean(rms_input_img_q**2, dim=-1, keepdim=True) + 1e-6)
    print(rrms)
    img_q = (rms_input_img_q * rrms) * q_img_rms_norm_scale
    return img_q.float(), q_img_rms_norm_scale.float()

img_q, rms_q_scale = get_output(rms_input_img_q)

with open(f'{TESTNAME}.txt', 'w') as f:
    rms_input_img_q_f = rms_input_img_q.to(torch.float32).flatten().detach().cpu().numpy()
    rms_q_scale_f = rms_q_scale.to(torch.float32).flatten().detach().cpu().numpy()
    img_q_f = img_q.to(torch.float32).flatten().detach().cpu().numpy()

    for i in trange(head_dim):
        f.write(repr(rms_q_scale_f[i]))
        if i < 3: print("2: ", repr(rms_q_scale_f[i]))
        f.write(' ')

    for i in trange(B*num_heads*img_in_dim*head_dim):
        f.write(repr(rms_input_img_q_f[i]))
        if i < 3: print("1: ", repr(rms_input_img_q_f[i]))

    for i in trange(B*num_heads*img_in_dim*head_dim):
        f.write(repr(img_q_f[i]))
        if i < 3: print("3: ", repr(img_q_f[i]))
        f.write(' ')