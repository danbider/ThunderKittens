import torch
import sys
import os
import time

import numpy as np

# These installs are for pulling in the TK source
# sys.path.append('../../../')
# from src.common.pyutils.test_build_utils import __eq
# sys.path.append('build/lib.linux-x86_64-cpython-311')

# try:
#     import hedgehog as mod
#     print(f"Succesfully imported hedgehog kernel")
# except:
#     print(f"Failed to import hedgehog kernel")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-3.10')
import hedgehog as tk

from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import median
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# try:
#     sys.path.append("../based/linear_attn_forward/")
#     from csrc.causal_dot_prod import causal_dot_product
#     print(f"Succesfully imported based kernel")
# except:
#     print(f"Failed to import based kernel")


def pytorch_test(dt, Q, K, V):
    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X
    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    o = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)
    return o


def torch_chunk_linear_attn(q, k, v, chunk_size=64):
    q = rearrange(q, 'b h (n c) d -> b h n c d', c = chunk_size) #* (q.shape[-1] **-0.5)
    k = rearrange(k, 'b h (n c) d -> b h n c d', c = chunk_size)
    v = rearrange(v, 'b h (n c) d -> b h n c d', c = chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([
        torch.zeros_like(kv[:, :, :1]),
        kv[:, :, :-1]
    ], dim=2)
    inter = q @ kv
    intra = ((q @ k.transpose(-1, -2)).masked_fill_(torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1), 0)) @ v
    o = inter + intra
    return rearrange(o, 'b h n c d -> b h (n c) d')


def torch_linear_attn(dt, q, k, v):
    q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
    kv_state = (k * v).cumsum(dim=2)
    out = (q * kv_state).sum(dim=-1)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    return out, last_kv_state


# def fast_transformer_test(dt, q, k, v):
#     y = causal_dot_product(
#         q.contiguous().to(dtype=torch.float32), 
#         k.contiguous().to(dtype=torch.float32),
#         v.contiguous().to(dtype=torch.float32),
#     )
#     return y

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time


def hedgehog_kernel_test(dt, q, k, v):
   
    b = Q.shape[0]
    h = Q.shape[1]
    n = Q.shape[2]
    
    d  = Q.shape[3]
    dv = V.shape[3]

    o   = torch.zeros_like(V)
    kv_state = torch.zeros((b, h, dv, d), dtype=dt, device='cuda')
    
    q_map = torch.cat([torch.exp(-Q), torch.exp(Q)], dim=-1)
    k_map = torch.cat([torch.exp(-K), torch.exp(K)], dim=-1)
    
    tk.hedgehog_based_tk(q, q_map, k, k_map, v, o, kv_state)

    return o, kv_state

def measure_performance_hh(b, h, n, d, dv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "CUDA not available"
    print("Using device:", device)
    
    q = torch.randn((b, h, n, d),  dtype=torch.bfloat16, device='cuda')
    k = torch.randn((b, h, n, d),  dtype=torch.bfloat16, device='cuda')
    v = torch.randn((b, h, n, dv), dtype=torch.bfloat16, device='cuda')
    
    q_map = torch.cat([torch.exp(-q), torch.exp(q)], dim=-1)
    k_map = torch.cat([torch.exp(-k), torch.exp(k)], dim=-1)
    
    q.grad = None
    k.grad = None
    v.grad = None
    
    o        = torch.zeros_like(v)
    kv_state = torch.zeros((b, h, dv, d), dtype=torch.bfloat16, device='cuda')
    
    o.grad        = None
    kv_state.grad = None
    
    torch.cuda.synchronize()
    
    # Warm up
    for _ in range(10):
        tk.hedgehog_based_tk(q, q_map, k, k_map, v, o, kv_state)
    
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    end_events   = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
    
    for i in range(20):
        # Timing the forward pass
        start_events[i].record()
        
        torch.cuda.synchronize()
        tk.hedgehog_based_tk(q, q_map, k, k_map, v, o, kv_state)
        torch.cuda.synchronize()
        
        end_events[i].record()
        
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    
    time_us = np.mean(times) * 1000
    
    print(f"Head Dim = {d}, Feature Dim = {dv}, Seq Len = {n}, Heads = {h}, Batch = {b}")
    print(f"Average time taken: {time_us:.2f} us")
    print("-" * 50)

def measure_performance_fa(b, h, n, d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "CUDA not available"
    print("Using device:", device)
    
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    ):
    
        q = torch.randn((b, h, n, d),  dtype=torch.bfloat16, device='cuda')
        k = torch.randn((b, h, n, d),  dtype=torch.bfloat16, device='cuda')
        v = torch.randn((b, h, n, d),  dtype=torch.bfloat16, device='cuda')
        
        o = torch.zeros_like(v)
        
        # warmup
        for _ in range(10):
            q.grad = None
            k.grad = None
            v.grad = None
            o.grad = None
            
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        # Prepare for timing
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        
        # Time the forward pass
        for i in range(30):
            start_events[i].record()
            torch.cuda.synchronize()
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    time_us = np.mean(times) * 1000
    print(f"Average time for forward pass in us: {time_us:.2f}")
    print(f"Average efficiency for forward pass in TFLOPS: {efficiency(flops(b, n, d, h, False, 'fwd'), time_us):.2f}")
    
    print("-" * 50)
    

B = 16
H = 16
N = 4096
D = 128
DV = 64

measure_performance_fa(B, H, N, D)
measure_performance_hh(B, H, N, D, DV)

# Q = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
# K = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
# V = torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')

# o, kv_state = hedgehog_kernel_test(torch.bfloat16, Q, K, V)

# def linear_attn_correct(dt):
#     b = 2
#     n = 1024
#     h = 16
#     d = 256
#     dv = 64
#     print(f"{b=}, {n=}, {d=}, {h=}")

#     Q   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
#     K   = torch.randn(b,h,n,d, dtype=dt, device='cuda')/d
#     V   = torch.randn(b,h,n,dv, dtype=dt, device='cuda')/dv

#     result_ref_1 = pytorch_test(dt, Q, K, V)
#     result_ref_2, kv_state_ref = torch_linear_attn(dt, Q, K, V)
#     result_ref_3 = torch_chunk_linear_attn(Q, K, V)
#     result_ref_4 = fast_transformer_test(dt, Q, K, V)
#     tk_result, tk_kv_state = hedgehog_kernel_test(dt, Q, K, V)

#     # Output and KV state
#     __eq(
#         "PyTorch Test v2 - Based Kernel Test", 
#         result_ref_2[0], 
#         tk_result[0], 
#         debug=False
#     )
#     __eq(
#         "PyTorch Test v2 - Based Kernel Test", 
#         kv_state_ref[0], 
#         tk_kv_state[0], 
#         debug=False
#     )

#     diff_state = torch.abs(kv_state_ref - tk_kv_state).max()
#     print(f"Max diff in state: {diff_state}")

#     # Output, more variants
#     __eq(
#         "PyTorch Test v1 - Based Kernel Test", 
#         result_ref_1[0], 
#         tk_result[0], 
#         debug=False
#     )
#     __eq(
#         "PyTorch Test v3 - Based Kernel Test", 
#         result_ref_3[0], 
#         tk_result[0], 
#         debug=False
#     )   
#     __eq(
#         "Fast Transformer Test - Based Kernel Test", 
#         result_ref_4[0], 
#         tk_result[0], 
#         debug=False
#     )


# print("Correctness test...")
# linear_attn_correct(torch.bfloat16)


