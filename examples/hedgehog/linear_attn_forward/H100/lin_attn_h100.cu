#define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#include "src/kittens.cuh"

#include <cooperative_groups.h>
#include <cuda/pipeline>

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

template<ducks::st::all ST>
__device__ inline void cumulative_add(ST &dst, const ST &inc) {
    // first do a reduction for each col
    constexpr int responsible_elements = (ST::cols + kittens::WARP_THREADS - 1) / kittens::WARP_THREADS;
    float acc[responsible_elements];

    // acc equal to the last row of dst
    for (auto i = 0; i < responsible_elements; i++) {
        auto col = (kittens::laneid() + (i * kittens::WARP_THREADS));
        if (col < dst.cols) {
            acc[i] = __bfloat162float(dst.data[(dst.rows - 1) * dst.cols + col]);
            __syncwarp();
            for (auto row = 0; row < dst.rows; row++) {
                acc[i] += __bfloat162float(inc.data[row * inc.cols + col]);
                dst.data[row * dst.cols + col] = __float2bfloat16(acc[i]);
            }
        }
        __syncwarp();
    }
}

#define ATTN_D 128
#define ATTN_F 256

#define tile_q_smem   st_bf<4, 4, wgmma_interleave_l>   
#define tile_k_smem   st_bf<4, 4, wgmma_interleave_l>  
#define tile_vo_smem  st_bf<4, 8, wgmma_interleave_l>    
#define tile_kv_smem  st_bf<4, 8, wgmma_interleave_l>

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_linear_attention(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, 
                                            CUtensorMap* tma_o,       CUtensorMap* tma_kv)
{
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    tile_q_smem  (&q_smem)   [4] = al.allocate<tile_q_smem,  4>(); // 32k
    tile_k_smem  (&k_smem)   [4] = al.allocate<tile_k_smem,  4>(); // 32k 
    tile_k_smem  (&k_c_smem) [4] = al.allocate<tile_k_smem,  4>(); // 32k
    tile_vo_smem (&v_smem)   [1] = al.allocate<tile_vo_smem, 1>(); // 16k
    tile_kv_smem (&kv_smem)      = al.allocate<tile_kv_smem   >();

    int warpid      = kittens::warpid(); 
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int tic = 0, toc = 1; 
    __shared__ uint64_t qkv_barrier; 

    int blocks = n / (kittens::TILE_DIM * 4); 

    rt_fl<1, 8> local_kv[4];
    #pragma unroll
    for (int rt = 0; rt < 4; rt++) {
        zero(local_kv[rt]); 
    }

    warpgroup::zero(k_c_smem[warpid]); 
    __syncthreads(); 

    col_vec<rt_fl<1, 4>> den_vec;

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl<1, 4>          local_attn; 
        rt_bf<1, 4>          local_attn_bf; 

        rt_fl<1, 4>          q_reg; 
        rt_fl<1, 4>          k_c_reg;

        rt_fl<1, 8>          local_o; 
        rt_bf<1, 8>          kv_bf[4];

        if (warpid == 0) {
            if (block == 0) {
            tma::init_barrier(qkv_barrier, 1); } 
            tma::set_bytes(qkv_barrier, size_bytes<tile_q_smem>*4 + size_bytes<tile_k_smem>*4 + size_bytes<tile_vo_smem>); 

            int tile_idx = (blockIdx.x * blocks) + block; 
            for (int i = 0; i < 4; i++) {
                tma::load_async(q_smem[i], tma_q, qkv_barrier, tile_idx, i); 
                tma::load_async(k_smem[i], tma_k, qkv_barrier, tile_idx, i); 
            }
            tma::load_async(v_smem[0], tma_v, qkv_barrier, tile_idx);
        }

        tma::arrive_and_wait(qkv_barrier, tic); 
        __syncthreads();

        zero(local_attn); 
        for (int j = 0; j < 4; j++) {
            warpgroup::mma_fence(local_attn); 
            warpgroup::mma_ABt(local_attn, q_smem[j], k_smem[j]); 
            warpgroup::mma_commit_group(); 
            warpgroup::mma_async_wait();
        }

        // now make causal
        for (int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(local_attn.tiles[0][j]);
            if (j > warpid) zero(attn_subtile);
            else if (j == warpid) make_causal(attn_subtile, attn_subtile, 0.0f);
        }
        __syncthreads();

        copy(local_attn_bf, local_attn); 
        warpgroup::mma_fence(local_o); 
        warpgroup::mm_AB(local_o, local_attn_bf, v_smem[0]); 
        warpgroup::mma_commit_group(); 
        warpgroup::mma_async_wait();

        for (auto rt = 0; rt < 4; rt++) {
            copy(kv_bf[rt], local_kv[rt]); 
            warpgroup::store(kv_smem, kv_bf[rt]); 
            __syncthreads();

            warpgroup::mma_fence(local_o); 
            warpgroup::mma_AB(local_o, q_smem[rt], kv_smem); 
            warpgroup::mma_commit_group();

            warpgroup::mma_fence(local_kv[rt]); 
            warpgroup::mma_AtB(local_kv[rt], k_smem[rt], v_smem[0]); 
            warpgroup::mma_commit_group(); 
            warpgroup::mma_async_wait();
        }

        __syncthreads();
        cumulative_add(k_c_smem[warpid], k_smem[warpid]);
        __syncthreads();

        zero(den_vec); 
        for (auto rt = 0; rt < 4; rt++) {
            warpgroup::load(q_reg, q_smem[rt]);
            warpgroup::load(k_c_reg, k_c_smem[rt]);

            mul(q_reg, q_reg, k_c_reg);
            row_sum(den_vec, q_reg, den_vec);
            __syncthreads(); 
        }
        
        div_row(local_o, local_o, den_vec);

        warpgroup::store(v_smem[0], local_o); 
        __syncthreads(); 

        if (warpid == 0) {
            int sidx = (blockIdx.x * blocks) + block; 
            tma::store_async(tma_o, v_smem[0], sidx); 
            tma::store_commit_group(); 
        }
        tma::store_async_wait(); 
        __syncthreads();
    }

    for (int rt = 0; rt < 4; rt++) {
        warpgroup::store(kv_smem, local_kv[rt]); 
        __syncthreads();

        if (warpid == 0) {
            int tile_idx = (blockIdx.x * 4) + rt; 
            tma::store_async(tma_kv, kv_smem, tile_idx); 
            tma::store_commit_group(); 
        }
        __syncthreads(); 
    } 

    tma::store_async_wait(); 
}

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void hh_lin_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor kv) {

    CHECK_INPUT(q); 
    CHECK_INPUT(k); 
    CHECK_INPUT(v); 
    CHECK_INPUT(kv); 
    CHECK_INPUT(o); 

    auto batch = q.size(0); 
    auto heads = q.size(1); 
    auto N     = q.size(2); 

    auto f     = q.size(3); 
    auto d     = v.size(3); 

    c10::BFloat16 *q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kv_ptr = kv.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr  =  o.data_ptr<c10::BFloat16>();

    const bf16* d_q        = reinterpret_cast<const bf16*>(q_ptr); 
    const bf16* d_k        = reinterpret_cast<const bf16*>(k_ptr);  
    const bf16* d_v        = reinterpret_cast<const bf16*>(v_ptr);  
    const bf16* d_kv_state = reinterpret_cast<bf16*>      (kv_ptr);  
    const bf16* d_o        = reinterpret_cast<bf16*>      (o_ptr);  

    CUtensorMap* tma_q_d  = tma::allocate_and_create_tensor_map<tile_q_smem> (d_q,        (batch*heads*N/(16 * 4)),     f/(16 * 4)); 
    CUtensorMap* tma_k_d  = tma::allocate_and_create_tensor_map<tile_k_smem> (d_k,        (batch*heads*N/(16 * 4)),     f/(16 * 4));
    CUtensorMap* tma_v_d  = tma::allocate_and_create_tensor_map<tile_vo_smem>(d_v,        (batch*heads*N/(16 * 4)),     d/(16 * 8));
    CUtensorMap* tma_o_d  = tma::allocate_and_create_tensor_map<tile_vo_smem>(d_o,        (batch*heads*N/(16 * 4)),     d/(16 * 8));
    CUtensorMap* tma_kv_d = tma::allocate_and_create_tensor_map<tile_kv_smem>(d_kv_state, (batch*heads*(d*2)/(16 * 4)), d/(16 * 8));

    unsigned long mem_size = kittens::MAX_SHARED_MEMORY;

    using T = kittens::bf16;
    using H = kittens::bf16;
    cudaFuncSetAttribute(
        hedgehog_linear_attention,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    hedgehog_linear_attention<<<batch*heads,NUM_THREADS,mem_size>>>(
        N, 
        tma_q_d, tma_k_d, tma_v_d, 
        tma_o_d, tma_kv_d
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
}

#else
#include "harness.impl"
#endif