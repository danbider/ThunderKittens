import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

# Individual linear layer functions
def linear_w1(dim, hidden_dim):
    return nn.Linear(dim, hidden_dim, bias=False).half().cuda()

def linear_w2(hidden_dim, dim):
    return nn.Linear(hidden_dim, dim, bias=False).half().cuda()

def linear_w3(dim, hidden_dim):
    return nn.Linear(dim, hidden_dim, bias=False).half().cuda()

def linear_wq(dim, n_heads, head_dim):
    return nn.Linear(dim, n_heads * head_dim, bias=False).half().cuda()

def linear_wk(dim, n_heads, head_dim):
    return nn.Linear(dim, n_heads * head_dim, bias=False).half().cuda()

def linear_wv(dim, n_heads, head_dim):
    return nn.Linear(dim, n_heads * head_dim, bias=False).half().cuda()

def linear_wo(n_heads, head_dim, dim):
    return nn.Linear(n_heads * head_dim, dim, bias=False).half().cuda()

# Define FeedForward using individual linear functions
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = linear_w1(dim, hidden_dim)
        self.w2 = linear_w2(hidden_dim, dim)
        self.w3 = linear_w3(dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Define ProjectionMatrices using individual linear functions
class ProjectionMatrices(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int):
        super().__init__()
        self.wq = linear_wq(dim, n_heads, head_dim)
        self.wk = linear_wk(dim, n_heads, head_dim)
        self.wv = linear_wv(dim, n_heads, head_dim)
        self.wo = linear_wo(n_heads, head_dim, dim)
        
def feedforward_benchmark(ff_layer, input_tensor):
    # Warm-up
    for _ in range(10):
        output = ff_layer(input_tensor)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    
    # Time the forward pass
    for i in range(100):
        start_events[i].record()
        torch.cuda.synchronize()
        output = ff_layer(input_tensor)
        torch.cuda.synchronize()
        end_events[i].record()
        
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    time_s = np.mean(times) / 1000
    
    # self.w1(x)
    tflop  = 2 * input_tensor.size(0) * input_tensor.size(1) * input_tensor.size(2) * ff_layer.w1.out_features / 1e12
    
    # self.w3(x)
    tflop += 2 * input_tensor.size(0) * input_tensor.size(1) * input_tensor.size(2) * ff_layer.w3.out_features / 1e12
    
    # F.silu(self.w1(x)) * self.w3(x)
    tflop += 6 * input_tensor.size(0) * input_tensor.size(1) * ff_layer.w1.out_features / 1e12
    
    # self.w2(F.silu(self.w1(x)) * self.w3(x))
    tflop += 2 * input_tensor.size(0) * ff_layer.w1.out_features * ff_layer.w2.out_features / 1e12
    
    tflpos = tflop / time_s
    
    return tflpos
        

# Benchmarking function for a single linear layer
def benchmark_linear_layer(layer, input_tensor):
    # Warm-up
    for _ in range(10):
        output = layer(input_tensor)
    
    # Prepare for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    # Time the forward pass
    for i in range(100):
        start_events[i].record()
        torch.cuda.synchronize()
        output = layer(input_tensor)
        torch.cuda.synchronize()
        end_events[i].record()
    
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    time_s = np.mean(times) / 1000
    tflop   = 2 * input_tensor.size(0) * input_tensor.size(1) * input_tensor.size(2) * layer.out_features / 1e12
    
    tflops  = tflop / time_s
    
    # reset gpu 
    torch.cuda.empty_cache()
    torch.cuda._sleep(10000)
    
    return tflops

def parameter_sweep():
    # Define parameter sweep ranges
    dims = [1024, 2048, 4096, 8192]
    batch_sizes = [1, 8, 16, 32, 64]
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    hidden_dims = [2048, 4096, 8192, 16384]
    n_heads_list = [1, 2, 4, 8, 16, 32]
    
    # Constant values for other parameters
    const_dim = 1024
    const_batch_size = 32
    const_seq_len = 2048
    const_hidden_dim = 8192
    const_n_heads = 32

    results = {}

    # Vary dims
    tflops = []
    for dim in dims:
        x = torch.randn((const_batch_size, const_seq_len, dim), dtype=torch.float16, device='cuda:0').requires_grad_()
        feedforward = FeedForward(dim, const_hidden_dim, dim // 4, None)
        projection_matrices = ProjectionMatrices(dim, dim // const_n_heads, const_n_heads)
        tflops.append({
            'feedforward_w1': benchmark_linear_layer(feedforward.w1, x),
            'feedforward_w2': benchmark_linear_layer(feedforward.w2, feedforward.w1(x)),
            'feedforward_w3': benchmark_linear_layer(feedforward.w3, x),
            'projection_wq': benchmark_linear_layer(projection_matrices.wq, x),
            'projection_wk': benchmark_linear_layer(projection_matrices.wk, x),
            'projection_wv': benchmark_linear_layer(projection_matrices.wv, x),
            'projection_wo': benchmark_linear_layer(projection_matrices.wo, projection_matrices.wq(x)),
            'feedforward': feedforward_benchmark(feedforward, x)
        })
    results['dims'] = (dims, tflops)
    print(f"Varying dims: {dims}, tflops: {tflops}")

    # Vary batch_sizes
    tflops = []
    for batch_size in batch_sizes:
        x = torch.randn((batch_size, const_seq_len, const_dim), dtype=torch.float16, device='cuda:0').requires_grad_()
        feedforward = FeedForward(const_dim, const_hidden_dim, const_dim // 4, None)
        projection_matrices = ProjectionMatrices(const_dim, const_dim // const_n_heads, const_n_heads)
        tflops.append({
            'feedforward_w1': benchmark_linear_layer(feedforward.w1, x),
            'feedforward_w2': benchmark_linear_layer(feedforward.w2, feedforward.w1(x)),
            'feedforward_w3': benchmark_linear_layer(feedforward.w3, x),
            'projection_wq': benchmark_linear_layer(projection_matrices.wq, x),
            'projection_wk': benchmark_linear_layer(projection_matrices.wk, x),
            'projection_wv': benchmark_linear_layer(projection_matrices.wv, x),
            'projection_wo': benchmark_linear_layer(projection_matrices.wo, projection_matrices.wq(x)),
            'feedforward': feedforward_benchmark(feedforward, x)
        })
    results['batch_sizes'] = (batch_sizes, tflops)
    print(f"Varying batch_sizes: {batch_sizes}, tflops: {tflops}")

    # Vary seq_lens
    tflops = []
    for seq_len in seq_lens:
        x = torch.randn((const_batch_size, seq_len, const_dim), dtype=torch.float16, device='cuda:0').requires_grad_()
        feedforward = FeedForward(const_dim, const_hidden_dim, const_dim // 4, None)
        projection_matrices = ProjectionMatrices(const_dim, const_dim // const_n_heads, const_n_heads)
        tflops.append({
            'feedforward_w1': benchmark_linear_layer(feedforward.w1, x),
            'feedforward_w2': benchmark_linear_layer(feedforward.w2, feedforward.w1(x)),
            'feedforward_w3': benchmark_linear_layer(feedforward.w3, x),
            'projection_wq': benchmark_linear_layer(projection_matrices.wq, x),
            'projection_wk': benchmark_linear_layer(projection_matrices.wk, x),
            'projection_wv': benchmark_linear_layer(projection_matrices.wv, x),
            'projection_wo': benchmark_linear_layer(projection_matrices.wo, projection_matrices.wq(x)),
            'feedforward': feedforward_benchmark(feedforward, x)
        })
    results['seq_lens'] = (seq_lens, tflops)
    print(f"Varying seq_lens: {seq_lens}, tflops: {tflops}")

    # Vary hidden_dims
    tflops = []
    for hidden_dim in hidden_dims:
        x = torch.randn((const_batch_size, const_seq_len, const_dim), dtype=torch.float16, device='cuda:0').requires_grad_()
        feedforward = FeedForward(const_dim, hidden_dim, const_dim // 4, None)
        projection_matrices = ProjectionMatrices(const_dim, const_dim // const_n_heads, const_n_heads)
        tflops.append({
            'feedforward_w1': benchmark_linear_layer(feedforward.w1, x),
            'feedforward_w2': benchmark_linear_layer(feedforward.w2, feedforward.w1(x)),
            'feedforward_w3': benchmark_linear_layer(feedforward.w3, x),
            'projection_wq': benchmark_linear_layer(projection_matrices.wq, x),
            'projection_wk': benchmark_linear_layer(projection_matrices.wk, x),
            'projection_wv': benchmark_linear_layer(projection_matrices.wv, x),
            'projection_wo': benchmark_linear_layer(projection_matrices.wo, projection_matrices.wq(x)),
            'feedforward': feedforward_benchmark(feedforward, x)
        })
    results['hidden_dims'] = (hidden_dims, tflops)
    print(f"Varying hidden_dims: {hidden_dims}, tflops: {tflops}")

    # Vary n_heads_list
    tflops = []
    for n_heads in n_heads_list:
        x = torch.randn((const_batch_size, const_seq_len, const_dim), dtype=torch.float16, device='cuda:0').requires_grad_()
        feedforward = FeedForward(const_dim, const_hidden_dim, const_dim // 4, None)
        projection_matrices = ProjectionMatrices(const_dim, const_dim // n_heads, n_heads)
        tflops.append({
            'feedforward_w1': benchmark_linear_layer(feedforward.w1, x),
            'feedforward_w2': benchmark_linear_layer(feedforward.w2, feedforward.w1(x)),
            'feedforward_w3': benchmark_linear_layer(feedforward.w3, x),
            'projection_wq': benchmark_linear_layer(projection_matrices.wq, x),
            'projection_wk': benchmark_linear_layer(projection_matrices.wk, x),
            'projection_wv': benchmark_linear_layer(projection_matrices.wv, x),
            'projection_wo': benchmark_linear_layer(projection_matrices.wo, projection_matrices.wq(x)),
            'feedforward': feedforward_benchmark(feedforward, x)
        })
    results['n_heads'] = (n_heads_list, tflops)
    print(f"Varying n_heads_list: {n_heads_list}, tflops: {tflops}")

    return results

def plot_results(results):
    for param, (values, tflops) in results.items():
        plt.figure(figsize=(10, 6))
        for key in tflops[0].keys():
            plt.plot(values, [tflop[key] for tflop in tflops], label=key)
        plt.xlabel(param)
        plt.ylabel('TFLOPs')
        plt.title(f'Llama 3 MLP Performance Sweep - varying {param}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'benchmark_varying_{param}.png')
        plt.close()

if __name__ == "__main__":
    results = parameter_sweep()
    plot_results(results)
