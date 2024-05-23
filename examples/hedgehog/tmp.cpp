# include "src/kittens.cuh"
#include <cuda/pipeline>

using namespace kittens;

using layout_q  = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_k  = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_v  = kittens::ducks::st_layout::wgmma_interleave;
using layout_o  = kittens::ducks::st_layout::wgmma_swizzle;
using layout_kv = kittens::ducks::st_layout::wgmma_interleave; 
using layout_kv2 = kittens::ducks::st_layout::wgmma_swizzle; 


#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

#define ATTN_D_QK 128 // hardcoded into this kernel
#define ATTN_D_VO 64  // hardcoded into this kernel

#define WINDOW_WIDTH (64)

#define tile_q_smem  st_bf_4x4<layout_q>
#define tile_k_smem  st_bf_4x4<layout_k>
#define tile_v_smem  st_bf_4x4<layout_v>
#define tile_o_smem  st_bf_4x4<layout_o>
#define tile_kv_smem st_bf_4x4<layout_kv>
#define tile_kv2_smem st_bf_4x4<layout_kv>

void diag();

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_attention(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_kv) {

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    tile_q_smem (&q_smem)[2][2] = al.allocate<tile_q_smem, 2, 2>(); // 16k * 2 (tictoc)
    tile_k_smem (&k_smem)[2][2] = al.allocate<tile_k_smem, 2, 2>(); // 16k * 2 (tictoc)
    tile_v_smem (&v_smem)[2]    = al.allocate<tile_v_smem, 2>(); // 8k * 2 (tictoc)
    tile_o_smem (&o_smem)       = al.allocate<tile_o_smem>(); // 8k

    // to hold 64 rows of q, k: 32k each = 65k
    // to hold 64 rows of v: 8k

    // tile_q_smem (&q_smem_fm)[4] = al.allocate<tile_q_smem, 4>(); // 8k * 4
    // tile_k_smem (&k_smem_fm)[4] = al.allocate<tile_k_smem, 4>(); // 8k * 4

    tile_kv_smem  (&kv_smem) = al.allocate<tile_kv_smem>(); // 8k
    tile_kv2_smem (&kv_smem_store)[4] = *reinterpret_cast<tile_kv2_smem(*)[4]>(q_smem[0]);

    if(threadIdx.x == 0) printf("%llu\n", (uint64_t(&kv_smem.data[0]) - uint64_t(&q_smem[0][0].data[0])) >> 10);

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int tic = 0, toc = 1;

    // launch tma loads
    __shared__ uint64_t qkv_barrier;

    int blocks = n / (64);

    if (threadIdx.x == 0) {
        tma::init_barrier(qkv_barrier, 1); // barrier for q and k
        tma::set_bytes(qkv_barrier, 2*size_bytes<tile_q_smem> + 2*size_bytes<tile_k_smem> + size_bytes<tile_v_smem>);
    }

    // launch tma loads 
    if (warpid == 0) {
        int tile_idx = (blockIdx.x * blocks);
        #pragma unroll
        for(int i = 0; i < 2; i++) {
            tma::load_async(q_smem[tic][i], tma_q, qkv_barrier, tile_idx, i);
            tma::load_async(k_smem[tic][i], tma_k, qkv_barrier, tile_idx, i);
        }
        tma::load_async(v_smem[tic], tma_v, qkv_barrier, tile_idx);
    }

    rt_fl_1x4<> local_kv[4];
    #pragma unroll
    for(int j = 0; j < 4; j++) {
        zero(local_kv[j]);
    }

    float last_norm, norm = 0;
    rt_fl_1x1<>::col_vec last_norm_vec, norm_vec;
    rt_fl_1x1<>::col_vec qk_diag;
    zero(norm_vec);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl_1x4 local_attn;
        rt_bf_1x4 local_attn_bf;
        rt_fl_1x4 local_o;

        tma::arrive_and_wait(qkv_barrier, tic);

        __syncthreads(); 
        if (warpid == 0) {
            tma::set_bytes(qkv_barrier, 2*size_bytes<tile_q_smem> + 2*size_bytes<tile_k_smem> + size_bytes<tile_v_smem>);
            if (block + 1 < blocks) {
                int tile_idx = (blockIdx.x * blocks) + (block + 1);
                #pragma unroll
                for(int i = 0; i < 2; i++) {
                    tma::load_async(q_smem[toc][i], tma_q, qkv_barrier, tile_idx, i);
                    tma::load_async(k_smem[toc][i], tma_k, qkv_barrier, tile_idx, i);
                }
                tma::load_async(v_smem[toc], tma_v, qkv_barrier, tile_idx);
            }
        }

        __syncthreads();
        // warpgroup::exp(q_smem_fm[0], q_smem[tic][0]);
        // warpgroup::exp(q_smem_fm[1], q_smem[tic][1]);
        // warpgroup::mul(q_smem_fm[2], q_smem[tic][0], __float2bfloat16(-1.f));
        // warpgroup::mul(q_smem_fm[3], q_smem[tic][1], __float2bfloat16(-1.f));
        // warpgroup::exp(q_smem_fm[2], q_smem_fm[2]);
        // warpgroup::exp(q_smem_fm[3], q_smem_fm[3]);
        // warpgroup::exp(k_smem_fm[0], k_smem[tic][0]);
        // warpgroup::exp(k_smem_fm[1], k_smem[tic][1]);
        // warpgroup::mul(k_smem_fm[2], k_smem[tic][0], __float2bfloat16(-1.f));
        // warpgroup::mul(k_smem_fm[3], k_smem[tic][1], __float2bfloat16(-1.f));
        // warpgroup::exp(k_smem_fm[2], k_smem_fm[2]);
        // warpgroup::exp(k_smem_fm[3], k_smem_fm[3]);
        __syncthreads();

        zero(local_attn);
        warpgroup::mma_fence(local_attn);
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            // warpgroup::mma_ABt(local_attn, q_smem[tic][j], k_smem[tic][j]); 
            // warpgroup::mma_ABt(local_attn, q_smem_fm[j], k_smem_fm[j]); 
            warpgroup::mma_commit_group(); 
        }
        warpgroup::mma_async_wait();
        exp(local_attn, local_attn);
        // copy old values
        // last_norm = norm;
        // copy(last_norm_vec, norm_vec);

        // // get the diagonal for the normalization
        // diag(qk_diag, reinterpret_cast<rt_bf_1x1<>&>(local_attn.tiles[0][warpid]));
        // // sum onto norm. this is now contains the linear norm for future iterations.
        // sum(norm, qk_diag, last_norm);

        // exp(local_attn, local_attn);
        // exp(qk_diag, qk_diag);

        // // add previous norm to get new normalization constants
        // add(norm_vec, qk_diag, norm);

        copy(local_attn_bf, local_attn); // now stored in bf16
        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_bf_1x1<>&>(local_attn_bf.tiles[0][j]);
            if (j>warpid) zero(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<bf16>::zero());
        }

        warpgroup::mma_fence(local_o);
        warpgroup::mm_AB(local_o, local_attn_bf, v_smem[tic]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();

        #pragma unroll
        for(int i = 0; i < 2; i++) {

            #pragma unroll
            for(int j = 0; j < 2; j++) {
            
                rt_bf_1x4<> q;
                rt_bf_4x1<> k;
                warpgroup::load(q, q_smem[tic][j]);
                auto k_subtile = subtile_inplace<4,1>(k_smem[tic][j], 0, warpid);
                load(k, k_subtile);
                rt_bf_1x4<> k_t;
                transpose_sep(k_t, k);
                if (i == 0) {
                    mul(q, q, -1.f);
                    mul(k_t, k_t, -1.f);
                }
                exp(q, q);
                exp(k_t, k_t);

                warpgroup::store(kv_smem, local_kv[2*i+j]);
                __syncthreads();

                warpgroup::mma_fence(local_o);
                warpgroup::mma_AB(local_o, q, kv_smem);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                warpgroup::mma_fence(local_kv[2*i+j]);
                warpgroup::mma_AB(local_kv[2*i+j], k_t, v_smem[tic]); // really AtB since k is transposed in advance
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                __syncthreads();
            }
        }

        tma::store_async_wait();
        warpgroup::store(o_smem, local_o);
        __syncthreads(); 

        // launch tma store for o
        if (warpid == 0) {
            int tile_idx = (blockIdx.x * blocks) + block;
            tma::store_async(tma_o, o_smem, tile_idx);
            tma::store_commit_group(); 
        }
    }
    #pragma unroll
    for(int j = 0; j < 4; j++) {
        warpgroup::store(kv_smem_store[j], local_kv[j]);
        __syncthreads(); 
        if(warpid == 0) {
            tma::store_async(tma_kv, kv_smem_store[j], blockIdx.x*4 + j);
            tma::store_commit_group();
        }
    }
    tma::store_async_wait();
}


#include "harness.impl"  // (comment out when using the code below)