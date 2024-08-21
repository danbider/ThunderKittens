#define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS_NORM (2) 
#define NUM_THREADS_NORM (NUM_WORKERS_NORM*kittens::WARP_THREADS)

#define B 1
#define H 24    
#define D 3072  
#define IMG_D 4080  
#define TXT_D 512
#define MULT_D 6
#define HEAD_D 128
const int d_model_tile = D / kittens::TILE_DIM;

using namespace kittens;
#define vec_smem_1xD sv_bf<d_model_tile>


__global__ __launch_bounds__(NUM_THREADS_NORM, 1)
void flux_prepare(
    const bf16* __img,
    const bf16* __vec,
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
    vec_smem_1xD (&img_s)      [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD (&residual_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD img_mod1_scale_s = al.allocate<vec_smem_1xD>(); 
    vec_smem_1xD img_mod1_shift_s = al.allocate<vec_smem_1xD>(); 

    // global loads
    if (warpid == 0) { 
        load(img_mod1_scale_s, img_mod1_scale_g);
        load(img_mod1_shift_s, img_mod1_shift_g);
        add(img_mod1_scale_s, img_mod1_scale_s, 1);
    }

    // pipelining
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.
 
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

        copy(residual_s[warpid][tic], img_s[warpid][tic]);
        sum(mean, residual_s[warpid][tic]);
        mean = mean / __float2bfloat16(D);
        sub(residual_s[warpid][tic], residual_s[warpid][tic], mean);  
        mul(img_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]);
        sum(var, img_s[warpid][tic]);
        var = var / __float2bfloat16(D);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-06f))));
        div(residual_s[warpid][tic], residual_s[warpid][tic], var);

        // compute norm
        mul(residual_s[warpid][tic], residual_s[warpid][tic], img_mod1_scale_s);
        add(residual_s[warpid][tic], residual_s[warpid][tic], img_mod1_shift_s);

        // save output
        store(o_g + (block*NUM_WORKERS_NORM +warpid)*D, residual_s[warpid][tic]); 
    }
}


// rms norm 
__global__ __launch_bounds__(NUM_THREADS_NORM, 1)
void flux_rmsnorm(
    const bf16* __img,
    const bf16* __rms_norm_scale,
    bf16* __o
) {

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    // shared memory setup to load from hbm
    const bf16 *img_g           = reinterpret_cast<const bf16*>(__img)+blockIdx.x*(IMG_D*D);
          bf16 *o_g             = reinterpret_cast<bf16*>(__o)+blockIdx.x*(IMG_D*D);
    const bf16 *rms_norm_scale_g = reinterpret_cast<const bf16*>(__rms_norm_scale);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    vec_smem_1xD (&img_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD (&residual_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xD,2,NUM_WORKERS_NORM>();
    vec_smem_1xD (&rms_norm_scale_s) = al.allocate<vec_smem_1xD>(); 

    // global loads
    if (warpid == 0) { 
        load(rms_norm_scale_s, rms_norm_scale_g);
    }

    // pipelining
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 rrms = __float2bfloat16(0.0f);
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
        }

        // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        copy(residual_s[warpid][tic], img_s[warpid][tic]);
        mul(img_s[warpid][tic], residual_s[warpid][tic], residual_s[warpid][tic]); // img_q**2
        sum(mean, img_s[warpid][tic]);
        mean = mean / __float2bfloat16(D); // get mean
        rrms = rrms + __float2bfloat16(1e-06f); // sqrt(rrms) + __float2bfloat16(1e-06f);

        // img_q = (img_q * rrms) * q_img_rms_norm_scale
        mul(residual_s[warpid][tic], residual_s[warpid][tic], rrms);
        mul(residual_s[warpid][tic], residual_s[warpid][tic], rms_norm_scale_s);

        // save output
        store(o_g + (block*NUM_WORKERS_NORM +warpid)*D, residual_s[warpid][tic]); 
    }
}


#include "harness.impl"
// #include "pointwise_gemm.cu"
