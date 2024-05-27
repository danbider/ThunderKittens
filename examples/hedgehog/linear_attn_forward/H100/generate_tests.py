import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 4096
D = 128

# accept arg for N 
N = int(sys.argv[1])

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)

def pytorch_test(Q, K, V): 
    
    Q = torch.cat([torch.exp(-Q.to(torch.bfloat16)).to(torch.float32), torch.exp(Q.to(torch.bfloat16)).to(torch.float32)], dim=-1)
    K = torch.cat([torch.exp(-K.to(torch.bfloat16)).to(torch.float32), torch.exp(K.to(torch.bfloat16)).to(torch.float32)], dim=-1)

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    out = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)
    
    K, V = K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (K * V).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    
    return out, last_kv_state

o, kv_state = pytorch_test(q, k, v)

print("-" * 80)
# print B, H, N, D
print(f'B = {B}')
print(f'H = {H}')
print(f'N = {N}')
print(f'D = {D}')

print("-" * 80)
# print desc of inputs and outputs
print(f'q: {q.shape} {q.dtype}')
print(f'k: {k.shape} {k.dtype}')
print(f'v: {v.shape} {v.dtype}')
print(f'o: {o.shape} {o.dtype}')
print(f'kv_state: {kv_state.shape} {kv_state.dtype}')

print("-" * 80)
with open(f'randn.txt', 'w') as f:
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    kv = kv_state.to(torch.float32).flatten().cpu().numpy()
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*D*2*D):
        f.write(repr(kv[i]))
        f.write(' ')

