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
H = 1
N = 4096
D = 128

# accept arg for N 
N = int(sys.argv[1])

torch.random.manual_seed(46)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

q = q/(float(D)**.5)
k = k/(float(D)**.5)
v = v/(float(D)**.5)

kv_state = (torch.zeros((B, H, D*2, D), dtype=torch.bfloat16, device='cuda'))
o = torch.zeros((B, H, N, D), dtype=torch.bfloat16, device='cuda')

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_state: torch.Tensor, o: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    causal = True
        
    tk_kernel.hh_lin_tk(q, k, v, o, kv_state)
        
    return o, kv_state, None

def quadratic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    q_max = torch.amax(q, dim=-1, keepdim=True)
    q_min = torch.amin(q, dim=-1, keepdim=True)
    q = torch.cat([
        torch.exp(q - q_max), torch.exp(-q + q_min)
    ], dim=-1)

    k_max = torch.amax(k, dim=-1, keepdim=True)
    k_min = torch.amin(k, dim=-1, keepdim=True)
    k = torch.cat([
        torch.exp(k - k_max), torch.exp(-k + k_min)
    ], dim=-1)
    
    y = None
    a = torch.einsum('bhmd,bhnd->bhmn', q, k)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    
    a = a / (a.sum(dim=-1, keepdim=True))
    
    y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    
    kv_state = (k.unsqueeze(-2) * v.unsqueeze(-1)).sum(dim=2)
    kv_state = kv_state.transpose(2, 3)
    
    return y, kv_state, None

linear_out, linear_kv_state, _    = linear_attention(q, k, v, kv_state, o)
quadratic_out, quadratic_kv_state, _ = quadratic_attention(q, k, v)

# print out 1/100 of avg mag of o and kv_state
print(f"1/100 of Avg mag of quadratic out: {torch.mean(torch.abs(quadratic_out)).item()/100}")

# print max diff
print(f"Max diff out: {torch.max(torch.abs(linear_out - quadratic_out)).item()}")
# print avg diff
print(f"Avg diff out: {torch.mean(torch.abs(linear_out - quadratic_out)).item()}")

# print out the last 10 elements
print(f"Linear out: {linear_out[0, 0, -10:, -5:]}")
print(f"Quadratic out: {quadratic_out[0, 0, -10:, -5:]}")

