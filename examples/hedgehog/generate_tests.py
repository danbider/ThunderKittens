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
    
torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)

def pytorch_test(Q, K, V, a, b, w): 
    
    Q_f = torch.concatenate([torch.exp(-Q.to(torch.bfloat16)).to(torch.float32), torch.exp(Q.to(torch.bfloat16)).to(torch.float32)], dim=-1)
    K_f = torch.concatenate([torch.exp(-K.to(torch.bfloat16)).to(torch.float32), torch.exp(K.to(torch.bfloat16)).to(torch.float32)], dim=-1)

    # create a block-diagonal matrix of size (Qc.shape[2]. Qc.shape[2]), where each block is 64x64
    n = Qc.shape[2]
    block_size = 64
    num_blocks = n // block_size
    mask = torch.block_diag(*[torch.ones(block_size, block_size) for _ in range(num_blocks)]).to('cuda').reshape((1,1,n,n))

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    # apply the mask to zero out off-diagonal blocks
    lin_ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Qc, Kc)*(1-mask))
    exp_ATT = make_causal(torch.exp(torch.einsum("bhnd,bhmd->bhnm", Qi, Ki))*mask)
    # norm = exp_ATT.max(dim=-1, keepdims=True)[0]

    ATT = lin_ATT + exp_ATT
    print(lin_ATT[0,0,0].cpu().tolist())
    print(lin_ATT[0,0,67].cpu().tolist())
    print(lin_ATT[0,0,127].cpu().tolist())
    ATT = ATT / ATT.sum(dim=-1, keepdims=True)
    out = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)
    
    K, V          = Kc.unsqueeze(-2), V.unsqueeze(-1)
    last_kv_state = (K * V).sum(dim=2).transpose(2, 3).to(torch.bfloat16).to(torch.float32)

    return Qi, Ki, Qc, Kc, out, last_kv_state

Q, K, Q_map, K_map, o, last_kv_state = pytorch_test(q, k, v, TESTNAME)

print(last_kv_state.shape)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf  = Q.to(torch.float32).flatten().cpu().numpy()
    kf  = K.to(torch.float32).flatten().cpu().numpy()
    vf  = v.to(torch.float32).flatten().cpu().numpy()
    of  = o.to(torch.float32).flatten().cpu().numpy()
    kvf = last_kv_state.to(torch.float32).flatten().cpu().numpy()
    
    q_map = Q_map.to(torch.float32).flatten().cpu().numpy()
    k_map = K_map.to(torch.float32).flatten().cpu().numpy()

    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*DV):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*DV):
        f.write(repr(of[i]))
        f.write(' ')
    for i in trange(B*H*D*DV*2):
        f.write(repr(kvf[i]))
        f.write(' ')
    for i in trange(B*H*N*D*2):
        f.write(repr(q_map[i]))
        f.write(' ')
    for i in trange(B*H*N*D*2):
        f.write(repr(k_map[i]))
        f.write(' ')


