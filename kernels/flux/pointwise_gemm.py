
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import thunderkittens as tk

num_iters = 100
warmup_iters = 100
dt = torch.bfloat16

class TorchGEMM(nn.Module):
    def __init__(self):
        super().__init__()   
    def forward(self, A, B):
        return torch.mm(F.silu(A), B)
        # return torch.mm(A, B)

class TKGEMM(nn.Module):
    def __init__(self, C):
        super().__init__() 
        self.C = C   
    def forward(self, A, B):
        return tk.pointwise_gemm(A, B, C)

# correctness
tile_size = 1024
A = torch.randn(tile_size, tile_size).to(dt).to("cuda")
B = torch.randn(tile_size, tile_size).to(dt).to("cuda")
C = torch.empty(tile_size, tile_size).to(dt).to("cuda")
tk.pointwise_gemm(A, B, C)
ref = torch.mm(F.silu(A), B)
ref = torch.mm(A * (1 / (1 + torch.exp(-A))), B)
print(C[0, 512:526])
print(ref[0, 512:526])
# print(ref_manual[0, :16])
diff = (C-ref).max()
print(f"Correctness diff: {diff}")

# # performance
# for tile_size in [1024, 2048, 4096]: 
#     A = torch.randn(tile_size, tile_size).to(dt).to("cuda")
#     B = torch.randn(tile_size, tile_size).to(dt).to("cuda")
#     C = torch.empty(tile_size, tile_size).to(dt).to("cuda")

#     for i in range(warmup_iters):
#         C = torch.mm(A, B)

#     torch_gemm = torch.compile(TorchGEMM())
#     tk_gemm = torch.compile(TKGEMM(C))

#     timings = []
#     timings_tk = []
#     for i in range(num_iters):

#         torch.cuda.synchronize()
#         start = time.time()
#         C_out = torch_gemm(A, B)
#         torch.cuda.synchronize()
#         end = time.time()
#         timings.append(end - start)

#         torch.cuda.synchronize()
#         start = time.time()
#         C_out_tk = tk_gemm(A, B)
#         torch.cuda.synchronize()
#         end = time.time()
#         timings_tk.append(end - start)
    
#     print(f"torch.mm: {sum(timings)/len(timings):.6f}")
#     print(f"tk  gemm: {sum(timings_tk)/len(timings_tk):.6f}")



