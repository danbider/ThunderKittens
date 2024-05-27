# include "src/kittens.cuh"
#include <cuda/pipeline>

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

using namespace kittens;

template<ducks::rt::row_layout RT>
__device__ static inline void wg_make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    const typename RT::dtype packed_val = base_types::packing<typename RT::dtype>::pack(val);
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            if(j < ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // below the diagonal, copy
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = src.tiles[i][j].data[k];
                }
            }
            else if(j > ((warpid() % kittens::WARPGROUP_WARPS) * dst.height) + i) { // above the diagonal, zero
                #pragma unroll
                for(int k = 0; k < dst.packed_per_tile; k++) {
                    dst.tiles[i][j].data[k] = packed_val;
                }
            }
            else { // on the diagonal, interesting!
                constexpr uint32_t MASK_X = 0xFF773311, MASK_Y = 0xF7733110; // magic numbers for on-diagonal core matrices
                dst.tiles[i][j].data[1] = src.tiles[i][j].data[1]; // below diagonal, copy
                dst.tiles[i][j].data[2] = packed_val; // above diagonal, zero
                if((MASK_X >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].x = src.tiles[i][j].data[0].x;
                    dst.tiles[i][j].data[3].x = src.tiles[i][j].data[3].x;
                }
                else {
                    dst.tiles[i][j].data[0].x = val;
                    dst.tiles[i][j].data[3].x = val;
                }
                if((MASK_Y >> laneid()) & 1) {
                    dst.tiles[i][j].data[0].y = src.tiles[i][j].data[0].y;
                    dst.tiles[i][j].data[3].y = src.tiles[i][j].data[3].y;
                }
                else {
                    dst.tiles[i][j].data[0].y = val;
                    dst.tiles[i][j].data[3].y = val;
                }
            }
        }
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

    tile_q_smem  (&q_smem) [4] = al.allocate<tile_q_smem,  4>(); // 32k
    tile_k_smem  (&k_smem) [4] = al.allocate<tile_k_smem,  4>(); // 32k 
    tile_vo_smem (&v_smem) [1] = al.allocate<tile_vo_smem, 1>(); // 16k
    tile_kv_smem (&kv_smem)[4] = al.allocate<tile_kv_smem, 4>(); // 65k

    int warpid      = kittens::warpid(); 
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int tic = 0, toc = 1; 
    __shared__ uint64_t qkv_barrier; 

    int blocks = n / (kittens::TILE_DIM * 4); 

    rt_fl<1, 8> local_kv[4];
    for (int rt = 0; rt < 4; rt++) {
        zero(local_kv[rt]); 
    }

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl<1, 4>          local_attn; 
        rt_bf<1, 4>          local_attn_bf; 

        rt_fl<1, 8>          local_o; 
        rt_bf<1, 8>          kv_bf[4];

        if (warpid == 0) {
            if (block == 0) {
            tma::init_barrier(qkv_barrier, 1); } 
            tma::set_bytes(qkv_barrier, size_bytes<tile_q_smem>*2 + size_bytes<tile_k_smem>*2 + size_bytes<tile_vo_smem>); 

            int tile_idx = (blockIdx.x * blocks) + block; 
            for (int i = 0; i < 2; i++) {
                tma::load_async(q_smem[i], tma_q, qkv_barrier, tile_idx, i); 
                tma::load_async(k_smem[i], tma_k, qkv_barrier, tile_idx, i); 
            }
            tma::load_async(v_smem[0], tma_v, qkv_barrier, tile_idx);
        }

        tma::arrive_and_wait(qkv_barrier, tic); 
        __syncthreads();

        // do in kernel feature map
        warpgroup::mul(q_smem[2], q_smem[0], -1.0f); 
        warpgroup::mul(q_smem[3], q_smem[1], -1.0f); 

        warpgroup::mul(k_smem[2], k_smem[0], -1.0f); 
        warpgroup::mul(k_smem[3], k_smem[1], -1.0f); 
        __syncthreads(); 

        for (int d = 0; d < 4; d++) {
            warpgroup::exp(q_smem[d], q_smem[d]); 
            warpgroup::exp(k_smem[d], k_smem[d]);
        }
        __syncthreads(); 

        zero(local_attn); 
        warpgroup::mma_fence(local_attn); 
        for (int j = 0; j < 4; j++) {
            warpgroup::mma_ABt(local_attn, q_smem[j], k_smem[j]); 
            warpgroup::mma_commit_group(); 
        }
        warpgroup::mma_async_wait();

        __syncthreads();
        // now make causal
        wg_make_causal(local_attn, local_attn, 0);
        __syncthreads();

        copy(local_attn_bf, local_attn); 
        warpgroup::mma_fence(local_o); 
        warpgroup::mm_AB(local_o, local_attn_bf, v_smem[0]); 
        warpgroup::mma_commit_group(); 
        warpgroup::mma_async_wait(); 

        for (auto rt = 0; rt < 4; rt++) {
            copy(kv_bf[rt], local_kv[rt]); 
            warpgroup::store(kv_smem[rt], kv_bf[rt]); 
            __syncthreads(); 

            warpgroup::mma_fence(local_o); 
            warpgroup::mma_AB(local_o, q_smem[rt], kv_smem[rt]); 
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); 
            __syncthreads();

            warpgroup::mma_fence(local_kv[rt]); 
            warpgroup::mma_AtB(local_kv[rt], k_smem[rt], v_smem[0]); 
            warpgroup::mma_commit_group(); 
            warpgroup::mma_async_wait(); 
            __syncthreads();
        }

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
        warpgroup::store(kv_smem[rt], local_kv[rt]); 
        __syncthreads();
    } 

    if (warpid == 0) {
        for (int rt = 0; rt < 4; rt++) {
            int tile_idx = (blockIdx.x * 4) + rt; 
            tma::store_async(tma_kv, kv_smem[rt], tile_idx); 
            tma::store_commit_group(); 
        } 
    }
    tma::store_async_wait(); 
}

#include "harness.impl"