import torch
from tqdm import trange
import numpy as np
import sys
import math

# can't PyPi this rn, so get from source: 
# pip install "grouped-query-attention-pytorch @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"
from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
N = int(sys.argv[1])
D = int(sys.argv[2])

H_QO = int(sys.argv[3])
H_KV = int(sys.argv[4])

torch.random.manual_seed(42)
q = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
k = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
v = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()

o, _ = scaled_dot_product_gqa(
    q.permute(0, 2, 1, 3).contiguous(),
    k.permute(0, 2, 1, 3).contiguous(),
    v.permute(0, 2, 1, 3).contiguous(),
    is_causal=True,
    need_weights=False,
)
o = o.permute(0, 2, 1, 3).contiguous()

if (H_QO == H_KV):
    o_ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    # compare outputs
    print("Difference: ", torch.abs(o - o_).max())

print("-------------------")
print("Q shape: ", q.shape)
print("K shape: ", k.shape)
print("V shape: ", v.shape)
print("O shape: ", o.shape)

# print out avg magnitude of output tensor
print(f'Average magnitude of output tensor: {o.abs().mean()}')
print(f'1/100 magnitude of output tensor: {o.abs().mean()/100}')
print("-------------------")

with open(f'randn_causal_gqa_{N}N_{D}D.txt', 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3]):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(k.shape[0] * k.shape[1] * k.shape[2] * k.shape[3]):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(v.shape[0] * v.shape[1] * v.shape[2] * v.shape[3]):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(o.shape[0] * o.shape[1] * o.shape[2] * o.shape[3]):
        f.write(repr(of[i]))
        f.write(' ')