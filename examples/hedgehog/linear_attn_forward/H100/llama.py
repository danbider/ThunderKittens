import torch
import sys
import os
from torch.cuda import Event
from torch.autograd import Function
import numpy as np

import math
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, project_root)
sys.path.append('build/lib.linux-x86_64-3.10')

import lin_attn_h100 as tk_kernel

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn

B = 1
H = 32
N = 4096
D = 128

# accept arg for N 
N = int(sys.argv[1])

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

q = q/(float(D))
k = k/(float(D))
v = v/(float(D))

kv_state = (torch.zeros((B, H, D*2, D), dtype=torch.bfloat16, device='cuda'))
o = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_state: torch.Tensor, o: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        tk_kernel.hh_lin_tk(q, k, v, o, kv_state)
        
        # q = torch.concatenate([torch.exp(q.to(torch.bfloat16)), torch.exp(-q.to(torch.bfloat16))], dim=-1)
        # k = torch.concatenate([torch.exp(k.to(torch.bfloat16)), torch.exp(-k.to(torch.bfloat16))], dim=-1)
        
        # o = o / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(dim=2)))[..., None]
        
        return o, None, None

def quadratic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None,
                        causal: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    q = torch.concatenate([torch.exp(q.to(torch.bfloat16)), torch.exp(-q.to(torch.bfloat16))], dim=-1)
    k = torch.concatenate([torch.exp(k.to(torch.bfloat16)), torch.exp(-k.to(torch.bfloat16))], dim=-1)
    
    y = None
    a = torch.einsum('bhmd,bhnd->bhmn', q, k)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    
    # Normalize to compute attention
    # a = a / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(dim=2)))[..., None]
    a = a / (a.sum(dim=-1, keepdim=True))
    
    if torch.isnan(a).sum() > 0:
        breakpoint()
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None

linear_out, _, _    = linear_attention(q, k, v, kv_state, o)
quadratic_out, _, _ = quadratic_attention(q, k, v)

# print out max error
print(f"Max error: {torch.max(torch.abs(linear_out - quadratic_out))}")

print(linear_out[0, 0, 1024:1034, :4])
print(quadratic_out[0, 0, 1024:1034, :4])

