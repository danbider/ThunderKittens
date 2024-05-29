import torch

b = 2
h = 2
l = 12
hd = 4

def make_causal(X):
    (b,h,n,d) = X.shape
    mask= ~(torch.arange(n).view(1,1,n,1) >= torch.arange(n).view(1,1,1,n)).expand(b,h,n,d)
    X[mask] = 0.
    return X

# setup
q = torch.randn(b, h, l, hd) * 2
k = torch.randn(b, h, l, hd) * 2
v = torch.randn(b, h, l, hd) * 2

# method
q2, k2, v2 = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)
kv_state = (k2 * v2).cumsum(dim=2)  # causal 
y2 = (q2 * kv_state).sum(dim=-1)
print(f"{y2.shape=}")

# method
cumsum_matrix = torch.tril(torch.ones((l, l)))
kv_state = torch.einsum("nm,bhmd,bhme->bhnde",cumsum_matrix,k,v)
y0   = torch.einsum("bhnd,bhnde->bhne", q, kv_state) 

# method
cumsum_matrix = torch.tril(torch.ones((l, l)))
A_qk = torch.einsum("bhnd,nm,bhmd->bhnmd", q, cumsum_matrix, k)
y3 = torch.einsum("bhnmd,bhme->bhne", A_qk, v)

# method
cumsum_matrix = torch.tril(torch.ones((l, l)))
A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k) * cumsum_matrix
y4 = torch.einsum("bhnm,bhme->bhne", A_qk, v)

# method
A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k)
A_qk = make_causal(A_qk)
y5 = torch.einsum("bhnm,bhme->bhne", A_qk, v)

# difference
print((y2 - y2).abs().max())
print((y2 - y3).abs().max())
print((y2 - y4).abs().max())
print((y3 - y4).abs().max())
print((y2 - y5).abs().max())