import torch
from tqdm import trange
import numpy as np
import sys

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
N = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])

TESTNAME = sys.argv[1]

print(f'Generating {TESTNAME} test with N={N}')

if TESTNAME == 'ones':
    torch.random.manual_seed(42)
    a = torch.ones((N, N), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((N, N), dtype=torch.bfloat16, device='cuda')
    c = torch.zeros((N, N), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'a_orientation':
    a = torch.arange(N*N, dtype=torch.bfloat16, device='cuda').reshape((N,N))
    b = torch.eye(N, dtype=torch.bfloat16, device='cuda')
    c = torch.zeros((N, N), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    a = torch.randn((N, N), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((N, N), dtype=torch.bfloat16, device='cuda')
    c = torch.zeros((N, N), dtype=torch.bfloat16, device='cuda')
# elif TESTNAME == 'qk_test':
#     q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
#     k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
#     v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
# elif TESTNAME == 'v_orientation':
#     q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
#     k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
#     v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
else:
    print('Invalid test name')
    sys.exit(0)

d = a@b.T + c

with open(f'{TESTNAME}_{N}.txt', 'w') as f:
    af = a.to(torch.float32).flatten().cpu().numpy()
    bf = b.to(torch.float32).flatten().cpu().numpy()
    cf = c.to(torch.float32).flatten().cpu().numpy()
    df = d.to(torch.float32).flatten().cpu().numpy()
    for i in trange(N*N):
        f.write(repr(af[i]))
        f.write(' ')
    for i in trange(N*N):
        f.write(repr(bf[i]))
        f.write(' ')
    for i in trange(N*N):
        f.write(repr(cf[i]))
        f.write(' ')
    for i in trange(N*N):
        f.write(repr(df[i]))
        f.write(' ')

print(f'Run the harness like `cat {TESTNAME}.txt | ./harness`')