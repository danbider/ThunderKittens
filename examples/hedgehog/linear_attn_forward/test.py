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

q = q/(float(D)**.5)
k = k/(float(D)**.5)
v = v/(float(D)**.5)

q_pt = q.clone()
k_pt = k.clone()
v_pt = v.clone()

o = torch.zeros(B, H, N, D, device='cuda', dtype=torch.bfloat16)

# divide q, k into 64 x 256 chunks
# divide v into 64 x 128 chunks
q = q.reshape(B, H, N//64, 64, D)
k = k.reshape(B, H, N//64, 64, D)
v = v.reshape(B, H, N//64, 64, D)
o = o.reshape(B, H, N//64, 64, D)

kv_state = torch.zeros(D*2, D, device='cuda', dtype=torch.bfloat16)
k_state  = torch.zeros(64, D*2, device='cuda', dtype=torch.bfloat16)

for block in range(N//64):
    q_block = q[:, :, block, :, :]
    k_block = k[:, :, block, :, :]
    v_block = v[:, :, block, :, :]
    
    # make q_block, k_block, v_block 2d (64, D)
    q_block = q_block.view(64, D)
    k_block = k_block.view(64, D)
    v_block = v_block.view(64, D)
    
    # apply feature map to q_block, k_block
    q_max = torch.amax(q_block, dim=-1, keepdim=True)
    q_min = torch.amin(q_block, dim=-1, keepdim=True)
    q_block = torch.cat([
        torch.exp(q_block - q_max), torch.exp(-q_block + q_min)
    ], dim=-1)
    
    k_max = torch.amax(k_block, dim=-1, keepdim=True)
    k_min = torch.amin(k_block, dim=-1, keepdim=True)
    k_block = torch.cat([
        torch.exp(k_block - k_max), torch.exp(-k_block + k_min)
    ], dim=-1)
    
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
kv_state = kv_state.unsqueeze(0).unsqueeze(0)
    

def pytorch_test(Q, K, V): 
    q_max = torch.amax(Q, dim=-1, keepdim=True)
    q_min = torch.amin(Q, dim=-1, keepdim=True)
    Q = torch.cat([
        torch.exp(Q - q_max), torch.exp(-Q + q_min)
    ], dim=-1)

    k_max = torch.amax(K, dim=-1, keepdim=True)
    k_min = torch.amin(K, dim=-1, keepdim=True)
    K = torch.cat([
        torch.exp(K - k_max), torch.exp(-K + k_min)
    ], dim=-1)

    
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

ans, pt_last_kv_state = pytorch_test(q_pt, k_pt, v_pt)

# print 1/100 of mag of ans
print("1/100 of avg mag of ans: ", torch.mean(torch.abs(ans)).item()/100)
print("1/100 of avg mag of kv_state: ", torch.mean(torch.abs(pt_last_kv_state)).item()/100)

# print avg diff between o and ans
print("avg diff out: ", torch.mean(torch.abs(o - ans)).item())
print("avg diff kv_state: ", torch.mean(torch.abs(kv_state - pt_last_kv_state)).item())

# print max diff between o and ans
print("o out: ", o[0, 0, -20:, :4])
print("ans out: ", ans[0, 0, -20:, :4])


# print new line
print("\n")
print("-" * 80)
    
    
#### CHUNKED SOFTMAX
torch.random.manual_seed(46)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

q = q/(float(D))
k = k/(float(D))
v = v/(float(D))

def pytorch_softmax(x):
    
    return torch.cat([
            torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
        ], dim=-1).clamp(min=1e-6)
    
def pytorch_softmax_gt(x):
    
    x_pos = x
    x_neg = -x
    
    x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
    x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
    
    # softmax(x) = torch.exp(x - x_max) / torch.sum(torch.exp(x - x_max), dim=-1) 
    
    x_pos = x_pos - x_pos_max
    x_neg = x_neg + x_neg_max
    
    x_pos_num = torch.exp(x_pos)
    x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
    
    x_neg_num = torch.exp(x_neg)
    x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
    
    x_pos = x_pos_num / x_pos_den
    x_neg = x_neg_num / x_neg_den
    
    x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
    
    return x

def TK_sim(Q, K):
    # goal: apply softmax feature map
    # input: Q, K of shape (B, H, N, D)
    # output: Q, K of shape (B, H, N, 2D)
    
    # chunk Q, K into 64 x 128
    Q = Q.reshape(B, H, N//64, 64, D)
    K = K.reshape(B, H, N//64, 64, D)
    
    Q_out = torch.zeros(B, H, N//64, 64, 2*D, device='cuda', dtype=torch.bfloat16)
    K_out = torch.zeros(B, H, N//64, 64, 2*D, device='cuda', dtype=torch.bfloat16)
    
    for block in range(N//64):
        Q_block = Q[:, :, block, :, :]
        K_block = K[:, :, block, :, :]
        
        Q_positive_block = Q_block.view(64, D)
        K_positive_block = K_block.view(64, D)
        
        ## computed in kernel
        Q_negative_block = -Q_block.view(64, D)
        K_negative_block = -K_block.view(64, D)
        
        # max
        q_pos_max = torch.amax(Q_positive_block, dim=-1, keepdim=True)
        q_neg_max = torch.amax(Q_negative_block, dim=-1, keepdim=True)
        
        k_pos_max = torch.amax(K_positive_block, dim=-1, keepdim=True)
        k_neg_max = torch.amax(K_negative_block, dim=-1, keepdim=True)
        
        # sub/add max for numerical stability
        q_pos = Q_positive_block - q_pos_max
        q_neg = Q_negative_block + q_neg_max
        
        k_pos = K_positive_block - k_pos_max
        k_neg = K_negative_block + k_neg_max
        
        # compute softmax
        q_pos_num = torch.exp(q_pos)
        q_pos_den = torch.sum(torch.exp(q_pos), dim=-1, keepdim=True)
        
        q_neg_num = torch.exp(q_neg)
        q_neg_den = torch.sum(torch.exp(q_neg), dim=-1, keepdim=True)
        
        k_pos_num = torch.exp(k_pos)
        k_pos_den = torch.sum(torch.exp(k_pos), dim=-1, keepdim=True)
        
        k_neg_num = torch.exp(k_neg)
        k_neg_den = torch.sum(torch.exp(k_neg), dim=-1, keepdim=True)
        
        Q_  = q_pos_num / q_pos_den
        Q_n = q_neg_num / q_neg_den
        
        K_  = k_pos_num / k_pos_den
        K_n = k_neg_num / k_neg_den
        
        Q_block = torch.cat([Q_, Q_n], dim=-1).clamp(min=1e-6)
        K_block = torch.cat([K_, K_n], dim=-1).clamp(min=1e-6)
        
        Q_out[:, :, block, :, :] = Q_block
        K_out[:, :, block, :, :] = K_block
        
    Q = Q_out.reshape(B, H, N, 2*D)
    K = K_out.reshape(B, H, N, 2*D)
    
    return Q, K
        

# print 1/100 of mag
print("1/100 of avg mag of q: ", torch.mean(torch.abs(q)).item()/100)
print("1/100 of avg mag of k: ", torch.mean(torch.abs(k)).item()/100)

q_gt = pytorch_softmax_gt(q)
k_gt = pytorch_softmax_gt(k)

q_tk, k_tk = TK_sim(q, k)

print("q TK check: ", torch.mean(torch.abs(q_gt - q_tk)).item())
print("k TK check: ", torch.mean(torch.abs(k_gt - k_tk)).item())




    
    