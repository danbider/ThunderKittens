# include "src/kittens.cuh"
#include <cuda/pipeline>

using namespace kittens;

// using layout = kittens::ducks::st_layout::swizzle;

// // sum of an array of tiles -- in fp32 to preserve maximal accuracy
// template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
// __device__ inline void tile_reduce(ST &dst, const ST (&src)[N_TILES]) {
//     constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
//     constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
//     float acc[RESPONSIBLE_ELEMENTS];
//     #pragma unroll
//     for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//         int idx = threadIdx.x + j*STRIDE;
//         if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst.data[idx]); // start
//     }
//     // then propagate accumulation through
//     for(int i = 0; i < N_TILES; i++) {
//         #pragma unroll
//         for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//             int idx = threadIdx.x + j*STRIDE;
//             if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(src[i].data[idx]); // accumulate
//         }
//     }
//     #pragma unroll
//     for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//         int idx = threadIdx.x + j*STRIDE;
//         if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst.data[idx] = __float2bfloat16(acc[j]); // set
//     }
// }


// // alternatively, sum onto the FIRST tile -- needed by attention.
// template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
// __device__ inline void tile_reduce(ST (&dst)[N_TILES]) {
//     constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
//     constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
//     float acc[RESPONSIBLE_ELEMENTS];
//     #pragma unroll
//     for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//         int idx = threadIdx.x + j*STRIDE;
//         if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst[0].data[idx]); // start
//     }
//     // then propagate accumulation through
//     for(int i = 1; i < N_TILES; i++) {
//         #pragma unroll
//         for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//             int idx = threadIdx.x + j*STRIDE;
//             if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(dst[i].data[idx]); // accumulate
//         }
//     }
//     #pragma unroll
//     for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
//         int idx = threadIdx.x + j*STRIDE;
//         if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst[0].data[idx] = __float2bfloat16(acc[j]); // set
//     }
// }

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

// sum of an array of tiles -- in fp32 to preserve maximal accuracy
template<int WORKERS, kittens::ducks::st::all ST, int N_TILES>
__device__ inline void tile_reduce(ST &dst, const ST (&src)[N_TILES]) {
    constexpr int STRIDE = WORKERS*kittens::WARP_THREADS;
    constexpr int RESPONSIBLE_ELEMENTS = (ST::num_elements+STRIDE-1) / STRIDE; // we know in advance this divides evenly.
    float acc[RESPONSIBLE_ELEMENTS];
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] = __bfloat162float(dst.data[idx]); // start
    }
    // then propagate accumulation through
    for(int i = 0; i < N_TILES; i++) {
        #pragma unroll
        for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
            int idx = threadIdx.x + j*STRIDE;
            if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) acc[j] += __bfloat162float(src[i].data[idx]); // accumulate
        }
    }
    #pragma unroll
    for(int j = 0; j < RESPONSIBLE_ELEMENTS; j++) {
        int idx = threadIdx.x + j*STRIDE;
        if(ST::num_elements%STRIDE == 0 || idx < ST::num_elements) dst.data[idx] = __float2bfloat16(acc[j]); // set
    }
}

using layout_q  = kittens::ducks::st_layout::wgmma_interleave; 
using layout_k  = kittens::ducks::st_layout::wgmma_interleave; 
using layout_v  = kittens::ducks::st_layout::wgmma_interleave;
using layout_o  = kittens::ducks::st_layout::wgmma_interleave;
using layout_kv = kittens::ducks::st_layout::wgmma_interleave; 


#define NUM_WORKERS (8)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

#define ATTN_D_QK 128 // hardcoded into this kernel
#define ATTN_D_VO 64  // hardcoded into this kernel

#define WINDOW_WIDTH (256)
static_assert(WINDOW_WIDTH%64==0 && WINDOW_WIDTH<=256);

#define tile_h 4

#define tile_w_qk (ATTN_D_QK/kittens::TILE_DIM) // 8
#define tile_w_vo (ATTN_D_VO/kittens::TILE_DIM) // 4

static_assert(tile_h % 4 == 0, "tile_h % 4 == 0, due to warpgrouped loads");

#define tile_q_smem  st_bf<tile_h,    tile_w_qk, layout_q>
#define tile_k_smem  st_bf<tile_h,    tile_w_qk, layout_k>
#define tile_v_smem  st_bf<tile_h,    tile_w_vo, layout_v>
#define tile_o_smem  st_bf<tile_h,    tile_w_vo, layout_o>
#define tile_kv_smem st_bf<tile_w_qk, tile_w_vo, layout_kv>

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_attention(int n, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o, CUtensorMap* tma_kv)
{

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    tile_q_smem (&q_smem)[2][NUM_WARPGROUPS] = al.allocate<tile_q_smem, 2, NUM_WARPGROUPS>(); // 32k * NUM_WARPGROUPS
    tile_k_smem (&k_smem)[2][NUM_WARPGROUPS] = al.allocate<tile_k_smem, 2, NUM_WARPGROUPS>(); // 32k * NUM_WARPGROUPS
    tile_v_smem (&v_smem)[2][NUM_WARPGROUPS] = al.allocate<tile_v_smem, 2, NUM_WARPGROUPS>(); // 16k * NUM_WARPGROUPS
    tile_o_smem (&o_smem)[NUM_WARPGROUPS] = al.allocate<tile_o_smem, NUM_WARPGROUPS>(); // 16k * NUM_WARPGROUPS

    tile_kv_smem (&kv_smem) = al.allocate<tile_kv_smem>(); // 32k

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int tic = 0; 
    int toc = 1; 

    // zero kv_smem
    if (warpid == 0) {
        zero(kv_smem);
    }

    // launch tma loads
    __shared__ uint64_t qk_barrier, v_barrier; 

    int qk_phase = 0, v_phase = 0;

    int blocks = n / (tile_h * NUM_WARPGROUPS * kittens::TILE_DIM);

    if (threadIdx.x == 0) {
        tma::init_barrier<tile_q_smem, NUM_WARPGROUPS * 2>(qk_barrier, 1); // barrier for q and k
        tma::init_barrier<tile_v_smem, NUM_WARPGROUPS    >(v_barrier , 1); // barrier for v
    }

    // launch tma loads 
    if (warpid == 0) {
        for (int wg = 0; wg < NUM_WARPGROUPS; wg++) {
            int tile_idx = (blockIdx.x * NUM_WARPGROUPS * blocks) + (0 * NUM_WARPGROUPS) + wg;
            tma::load_async(q_smem[tic][wg], tma_q, qk_barrier, tile_idx);
            tma::load_async(k_smem[tic][wg], tma_k, qk_barrier, tile_idx);
            tma::load_async(v_smem[tic][wg], tma_v, v_barrier , tile_idx);
        }
    }

    rt_fl<tile_w_qk/kittens::WARPGROUP_WARPS, tile_w_vo> local_kv;
    zero(local_kv);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_h> local_attn;
        rt_bf<tile_h/kittens::WARPGROUP_WARPS, tile_h> local_attn_bf;
        rt_fl<tile_h/kittens::WARPGROUP_WARPS, tile_w_vo> local_o;

        tma::arrive_and_wait(qk_barrier, qk_phase);
        tma::arrive_and_wait(v_barrier , v_phase );

        qk_phase ^= 1;
        v_phase  ^= 1;

        __syncthreads(); 
        if (warpid == 0) {
            tma::set_bytes(qk_barrier, 2 * NUM_WARPGROUPS * q_smem[0][0].num_elements * sizeof(bf16));
            tma::set_bytes(v_barrier,      NUM_WARPGROUPS * v_smem[0][0].num_elements * sizeof(bf16));

            if (block + 1 < blocks) {
                for (int wg = 0; wg < NUM_WARPGROUPS; wg++) {
                    int tile_idx = (blockIdx.x * NUM_WARPGROUPS * blocks) + ((block + 1) * NUM_WARPGROUPS) + wg;
                    tma::load_async(q_smem[toc][wg], tma_q, qk_barrier, tile_idx);
                    tma::load_async(k_smem[toc][wg], tma_k, qk_barrier, tile_idx);
                    tma::load_async(v_smem[toc][wg], tma_v, v_barrier , tile_idx);
                }
            }
        }

        warpgroup::mma_fence(local_attn);
        warpgroup::mm_ABt(local_attn, q_smem[tic][warpgroupid], k_smem[tic][warpgroupid]); 
        warpgroup::mma_commit_group(); 

        warpgroup::mma_async_wait();

        wg_make_causal(local_attn, local_attn, kittens::base_types::constants<bf16>::zero());

        copy(local_attn_bf, local_attn);
        zero(local_o);
        warpgroup::mma_fence(local_o);
        warpgroup::mma_AB(local_o, local_attn_bf, v_smem[tic][warpgroupid]);
        warpgroup::mma_commit_group();

        warpgroup::mma_async_wait();

        if (block > 0) {
            warpgroup::store(kv_smem, local_kv);
            __syncthreads();
        }

        warpgroup::mma_fence(local_o);
        warpgroup::mma_AB(local_o, q_smem[tic][warpgroupid], kv_smem);
        warpgroup::mma_commit_group();

        if (block > 0) {
            tma::store_async_wait(); 
        }

        warpgroup::mma_async_wait();

        warpgroup::store(o_smem[warpgroupid], local_o);
        __syncthreads(); 

        warpgroup::mma_fence(local_kv);
        warpgroup::mma_AtB(local_kv, k_smem[tic][warpgroupid], v_smem[tic][warpgroupid]);
        warpgroup::mma_commit_group();

        // launch tma store for o
        if (warpid % 4 == 0) {
            int tile_idx = (blockIdx.x * NUM_WARPGROUPS * blocks) + (block * NUM_WARPGROUPS) + warpgroupid;
            tma::store_async(tma_o, o_smem[warpgroupid], tile_idx);
            tma::store_commit_group(); 
        }
    }

    __syncthreads(); 
    if (warpid == 0) {
        // kv state store
        tma::store_async(tma_kv, kv_smem, blockIdx.x);
        tma::store_commit_group();
    }

    tma::store_async_wait();
}


#include "harness.impl"  // (comment out when using the code below)