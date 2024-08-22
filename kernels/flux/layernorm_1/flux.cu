// #define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS_NORM (2) 
#define NUM_THREADS_NORM (NUM_WORKERS_NORM*kittens::WARP_THREADS)

const int D = 3072;
const int IMG_D = 4080; 
const int d_model_tile = D / kittens::TILE_DIM;

using namespace kittens;
#define vec_smem_1xD sv_bf<d_model_tile>

__global__ __launch_bounds__(NUM_THREADS_NORM, 1)
void flux_prepare(
    const bf16* __img,
    const bf16* __img_mod1_shift,
    const bf16* __img_mod1_scale,
    bf16* __o
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *img_g           = reinterpret_cast<const bf16*>(__img)+blockIdx.x*(IMG_D*D);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)+blockIdx.x*(IMG_D*D);
    const bf16 *img_mod1_scale_g = reinterpret_cast<const bf16*>(__img_mod1_scale);
    const bf16 *img_mod1_shift_g = reinterpret_cast<const bf16*>(__img_mod1_shift);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    vec_smem_1xD (&img_s)     [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD (&scratch_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD (&img_mod1_scale_s) = al.allocate<vec_smem_1xD>(); 
    vec_smem_1xD (&img_mod1_shift_s) = al.allocate<vec_smem_1xD>(); 

    // pipelining
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.

    // global loads
    if (warpid == 0) { 
        load(img_mod1_scale_s, img_mod1_scale_g);
        load(img_mod1_shift_s, img_mod1_shift_g);
        add(img_mod1_scale_s, img_mod1_scale_s, __float2bfloat16(1.0f));
    }
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      
    
    load_async(img_s[warpid][tic], img_g + warpid*D, D, barrier_cheat);
    __syncthreads();
    
    int n_blocks = IMG_D / NUM_WORKERS_NORM;
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            auto next_idx = (block + 1)*NUM_WORKERS_NORM + warpid; 
            load_async(img_s[warpid][toc], img_g + next_idx*D, D, barrier_cheat);
            __syncthreads();
        }

        copy(scratch_s[warpid][tic], img_s[warpid][tic]);
        sum(mean, scratch_s[warpid][tic]);
        mean = mean / __float2bfloat16(D);
        sub(scratch_s[warpid][tic], scratch_s[warpid][tic], mean);  
        mul(img_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]);
        sum(var, img_s[warpid][tic]);
        var = var / __float2bfloat16(D);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-06f))));
        div(scratch_s[warpid][tic], scratch_s[warpid][tic], var);
        __syncthreads();

        // compute norm
        mul(scratch_s[warpid][tic], scratch_s[warpid][tic], img_mod1_scale_s);
        add(scratch_s[warpid][tic], scratch_s[warpid][tic], img_mod1_shift_s);

        // save output
        store(o_g + (block*NUM_WORKERS_NORM +warpid)*D, scratch_s[warpid][tic]); 
    }
}

// #ifdef TORCH_COMPILE
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void 
fused_flux_layernorm(
    const torch::Tensor x, 
    const torch::Tensor shift, 
    const torch::Tensor scale, 
    torch::Tensor out
) {
    CHECK_INPUT(x);
    CHECK_INPUT(shift);
    CHECK_INPUT(scale);
    CHECK_INPUT(out);

    int batch = x.size(0);
    int seq_dim = x.size(1);
    int model_dim = x.size(2);

    TORCH_CHECK(batch == out.size(0),    "Differing batch sizes for input and output?");
    TORCH_CHECK(seq_dim == out.size(1),  "Differing seq len for input and output?");
    TORCH_CHECK(model_dim == scale.size(0),  "Differing dim for input and scale?");
    TORCH_CHECK(model_dim == shift.size(0),  "Differing dim for input and shift?");

    TORCH_CHECK(seq_dim % kittens::TILE_DIM == 0,  "sequence length is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *x_ptr     = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *shift_ptr = shift.data_ptr<c10::BFloat16>();
    c10::BFloat16 *scale_ptr = scale.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr     = out.data_ptr<c10::BFloat16>();

    const bf16* d_x = reinterpret_cast<const bf16*>(x_ptr);
    const bf16* d_shift  = reinterpret_cast<const bf16*>(shift_ptr);
    const bf16* d_scale  = reinterpret_cast<const bf16*>(scale_ptr);
          bf16* d_o      = reinterpret_cast<bf16*>(o_ptr);

    // launch variables
    unsigned long mem_size = 95000;

    cudaFuncSetAttribute(
        flux_prepare,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    flux_prepare<<<batch,NUM_THREADS_NORM,mem_size>>>(
        d_x, d_shift, d_scale, d_o
    );  

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// #else
// #include "harness.impl"
// #endif
