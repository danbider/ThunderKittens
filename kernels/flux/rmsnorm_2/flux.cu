#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS_NORM (2) 
#define NUM_THREADS_NORM (NUM_WORKERS_NORM*kittens::WARP_THREADS)

const int H = 24;    
const int D = 3072;  
const int IMG_D = 4080;  
const int HEAD_D = 128;
const int d_head_tile = HEAD_D / kittens::TILE_DIM;

using namespace kittens;
#define vec_smem_1xHEAD_D sv_bf<d_head_tile>

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
    const bf16 *img_g            = reinterpret_cast<const bf16*>(__img)+blockIdx.x*(IMG_D*HEAD_D);
          bf16 *o_g              = reinterpret_cast<bf16*>(__o)+blockIdx.x*(IMG_D*HEAD_D);
    const bf16 *rms_norm_scale_g = reinterpret_cast<const bf16*>(__rms_norm_scale);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    vec_smem_1xHEAD_D (&img_s)     [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    vec_smem_1xHEAD_D (&scratch_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    vec_smem_1xHEAD_D (&rms_norm_scale_s) = al.allocate<vec_smem_1xHEAD_D>(); 

    // global loads
    if (warpid == 0) { load(rms_norm_scale_s, rms_norm_scale_g); }

    // pipelining
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.
 
    bf16 mean = __float2bfloat16(0.0f);
    bf16 rrms = __float2bfloat16(0.0f);
    bf16 var  = __float2bfloat16(0.0f);      

    load_async(img_s[warpid][tic], img_g + warpid*HEAD_D, HEAD_D, barrier_cheat);
    __syncthreads();
    
    int n_blocks = IMG_D / NUM_WORKERS_NORM;
    if (threadIdx.x == 0 && blockIdx.x == 0) { 
            printf("n_blocks=%d\n", n_blocks);
        }
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            auto next_idx = (block + 1)*NUM_WORKERS_NORM + warpid; 
            load_async(img_s[warpid][toc], img_g + next_idx*HEAD_D, HEAD_D, barrier_cheat);
        }

        // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        copy(scratch_s[warpid][tic], img_s[warpid][tic]);
        mul(img_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]); // img_q**2
        sum(mean, img_s[warpid][tic]);
        if (threadIdx.x == 0 && block==0 && blockIdx.x == 0) { 
            printf("mean_init=%f\n", __bfloat162float(mean));
        }
        mean = mean / __float2bfloat16(HEAD_D);
        if (threadIdx.x == 0 && block==0  && blockIdx.x == 0) { 
            printf("mean_start=%f\n", __bfloat162float(mean));
            printf("d_head_tile=%d\n", d_head_tile);
        }
        rrms = __float2bfloat16(1 / sqrt(__bfloat162float(mean + __float2bfloat16(1e-06f))));

        // img_q = (img_q * rrms) * q_img_rms_norm_scale;
        mul(scratch_s[warpid][tic], scratch_s[warpid][tic], rrms);
        mul(scratch_s[warpid][tic], scratch_s[warpid][tic], rms_norm_scale_s);

        if (threadIdx.x == 0 && block==0  && blockIdx.x == 0) { 
            printf("mean_end=%f\n", __bfloat162float(mean));
            printf("rrms_end=%f\n", __bfloat162float(rrms));
            printf("head=%d\n", HEAD_D);
        }

        // save output
        store(o_g + (block*NUM_WORKERS_NORM +warpid)*HEAD_D, scratch_s[warpid][tic]); 
        __syncthreads();
    }
}

#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void 
fused_flux_rmsnorm(
    const torch::Tensor img_in, 
    const torch::Tensor rms_scale, 
    torch::Tensor out
) {
    CHECK_INPUT(img_in);
    CHECK_INPUT(rms_scale);
    CHECK_INPUT(out);

    int batch = img_in.size(0);
    int heads = img_in.size(1);
    int seq_dim = img_in.size(2);
    int head_dim = img_in.size(3);

    TORCH_CHECK(batch == out.size(0),    "Differing batch sizes for input and output?");
    TORCH_CHECK(seq_dim == out.size(2),  "Differing seq len for input and output?");
    TORCH_CHECK(head_dim == rms_scale.size(0),  "Differing head_dim for input and rms_scale?");

    TORCH_CHECK(seq_dim % kittens::TILE_DIM == 0,  "sequence length is divisible by 16?");
    TORCH_CHECK(head_dim % kittens::TILE_DIM == 0, "head dimension is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *rms_in_img_q_ptr  = img_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *rms_q_scale_ptr   = rms_scale.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr             = out.data_ptr<c10::BFloat16>();

    const bf16* d_rms_in_img_q = reinterpret_cast<const bf16*>(rms_in_img_q_ptr);
    const bf16* d_rms_q_scale  = reinterpret_cast<const bf16*>(rms_q_scale_ptr);
          bf16* d_o            = reinterpret_cast<bf16*>(o_ptr);

    // launch variables
    unsigned long mem_size = 8000;

    cudaFuncSetAttribute(
        flux_rmsnorm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    flux_rmsnorm<<<batch*heads,NUM_THREADS_NORM,mem_size>>>(
        d_rms_in_img_q, d_rms_q_scale, d_o
    );  

    CHECK_CUDA_ERROR(cudaGetLastError());
}

// #include "harness.impl"
