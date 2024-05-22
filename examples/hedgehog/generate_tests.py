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
DV = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
elif TESTNAME in ['twos']:
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')*2).to(torch.float32)
    v = (torch.ones((B, H, N, DV), dtype=torch.bfloat16, device='cuda')).to(torch.float32)
elif TESTNAME in ['arange']:
    q = (torch.ones(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV)
    k = (torch.arange(B*H*N*D, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, D)).to(torch.float32)/(D*DV*2)
    v = (torch.ones(B*H*N*DV, dtype=torch.bfloat16, device='cuda').reshape(B, H, N, DV)).to(torch.float32)
elif TESTNAME in ['randn']:
    torch.random.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/float(D)**.5).to(torch.float32)
    v = (torch.randn((B, H, N, DV), dtype=torch.bfloat16, device='cuda')/DV).to(torch.float32)
else:
    print('Invalid test name')
    sys.exit(0)

def pytorch_test(Qi, Ki, V, TESTNAME='all'):
    
    Q = torch.concatenate([torch.exp(-Qi.to(torch.bfloat16)).to(torch.float32), torch.exp(Qi.to(torch.bfloat16)).to(torch.float32)], dim=-1)
    K = torch.concatenate([torch.exp(-Ki.to(torch.bfloat16)).to(torch.float32), torch.exp(Ki.to(torch.bfloat16)).to(torch.float32)], dim=-1)

    def make_causal(X):
        (b,h,n,m) = X.shape
        mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,n)
        X[mask] = 0.
        return X

    ATT = make_causal(torch.einsum("bhnd,bhmd->bhnm", Q, K))
    out = torch.einsum("bhnm,bhmd->bhnd", ATT, V).to(torch.bfloat16)
    
    K, V          = K.unsqueeze(-2), V.unsqueeze(-1)
    last_kv_state = (K * V).sum(dim=2).transpose(2, 3).to(torch.bfloat16).to(torch.float32)

    return Qi, Ki, out, last_kv_state

Q, K, o, last_kv_state = pytorch_test(q, k, v, TESTNAME)

print(last_kv_state.shape)

with open(f'{TESTNAME}.txt', 'w') as f:
    qf  = Q.to(torch.float32).flatten().cpu().numpy()
    kf  = K.to(torch.float32).flatten().cpu().numpy()
    vf  = v.to(torch.float32).flatten().cpu().numpy()
    of  = o.to(torch.float32).flatten().cpu().numpy()
    kvf = last_kv_state.to(torch.float32).flatten().cpu().numpy()

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


