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

torch.random.manual_seed(46)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

q = q/(float(D))
k = k/(float(D))
v = v/(float(D))

q = torch.cat([
    torch.softmax(q, dim=-1), torch.softmax(-q, dim=-1)
], dim=-1).clamp(min=1e-6)

k = torch.cat([
    torch.softmax(k, dim=-1), torch.softmax(-k, dim=-1)
], dim=-1).clamp(min=1e-6)

def pytorch_test(Q, K, V): 
    
    causal = True
    
    a = torch.einsum('bhmd,bhnd->bhmn', Q, K)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    
    # Normalize to compute attention
    # a = a / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(dim=2)))[..., None]
    a = a / (a.sum(dim=-1, keepdim=True))
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    K, V = K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (K * V).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    
    return out, last_kv_state

o, kv_state = pytorch_test(q, k, v)

avg_o = torch.mean(torch.abs(o))
avg_kv = torch.mean(torch.abs(kv_state))

print(f"1/100 of Avg mag of o: {avg_o.item()/100}")
print(f"1/100 of Avg mag of kv: {avg_kv.item()/100}")

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
    for i in trange(B*H*N*D*2):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D*2):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*2*D*D):
        f.write(repr(kv[i]))
        f.write(' ')

