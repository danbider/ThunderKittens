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

torch.random.manual_seed(46)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

q = q/(float(D))
k = k/(float(D))
v = v/(float(D))

q_max = torch.amax(q, dim=-1, keepdim=True)
q = torch.cat([
    torch.exp(q - q_max), torch.exp(-q + q_max)
], dim=-1)

k_max = torch.amax(k, dim=-1, keepdim=True)
k = torch.cat([
    torch.exp(k - k_max), torch.exp(-k + k_max)
], dim=-1)

q_pt = q.clone()
k_pt = k.clone()
v_pt = v.clone()

o = torch.zeros(B, H, N, D, device='cuda', dtype=torch.bfloat16)

# divide q, k into 64 x 256 chunks
# divide v into 64 x 128 chunks
q = q.reshape(B, H, N//64, 64, D*2)
k = k.reshape(B, H, N//64, 64, D*2)
v = v.reshape(B, H, N//64, 64, D)
o = o.reshape(B, H, N//64, 64, D)

kv_state = torch.zeros(D*2, D, device='cuda', dtype=torch.bfloat16)
k_state  = torch.zeros(64, D*2, device='cuda', dtype=torch.bfloat16)

for block in range(N//64):
    q_block = q[:, :, block, :, :]
    k_block = k[:, :, block, :, :]
    v_block = v[:, :, block, :, :]
    
    # make q_block, k_block, v_block 2d (64, 256), (64, 256), (64, 128)
    q_block = q_block.view(64, D*2)
    k_block = k_block.view(64, D*2)
    v_block = v_block.view(64, D)
    
    local_attn = torch.einsum('nf,mf->nm', q_block, k_block)
    # make local_attn causal
    m, n = local_attn.shape[-2:]
    causal_mask = torch.ones((m, n), device = local_attn.device, dtype = torch.bool).triu(n - m + 1)
    local_attn = local_attn.masked_fill(causal_mask, 0)
    # multiply by v_block
    local_o = torch.einsum('mn,nd->md', local_attn, v_block)
    
    # add in q * kv_state to local_o
    local_o += torch.einsum('mf,fd->md', q_block, kv_state)
    
    # update kv_state
    kv_state = kv_state + torch.einsum('nf,nd->fd', k_block, v_block)
    
    # # update k_state (cumulative sum of k_block)
    k_state = k_state[-1, :]
    k_state = k_state + k_block.cumsum(dim=0)
    
    # # convert diagonal of square local_attn to vec
    norm_vec = (q_block * k_state).sum(dim=-1, keepdim=True)
    
    # normalize local_o by norm_vec
    local_o = local_o / norm_vec
    
    # store local_o in o
    o[:, :, block, :, :] = local_o

o = o.reshape(B, H, N, D)
    

def pytorch_test(Q, K, V): 
    
    causal = True
    
    a = torch.einsum('bhmd,bhnd->bhmn', Q, K)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    
    # Normalize to compute attention
    # a = a / (torch.einsum("bhld,bhld->bhl", Q, K.cumsum(dim=2)))[..., None]
    a = a / (a.sum(dim=-1, keepdim=True))
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    K, V = K.unsqueeze(-2), V.unsqueeze(-1)
    kv_state = (K * V).cumsum(dim=2)
    last_kv_state = kv_state[:, :, -1].transpose(2, 3)
    
    return out, last_kv_state

ans, _ = pytorch_test(q_pt, k_pt, v_pt)

# print 1/100 of mag of ans
print("1/100 of avg mag of ans: ", torch.mean(torch.abs(ans)).item()/100)

# print avg diff between o and ans
print("avg diff: ", torch.mean(torch.abs(o - ans)).item())

# print max diff between o and ans
print("o out: ", o[0, 0, -10:, :4])
print("ans out: ", ans[0, 0, -10:, :4])
    

    
    