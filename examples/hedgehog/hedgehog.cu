# include "src/kittens.cuh"
#include <cuda/pipeline>

using namespace kittens;

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
#define NUM_WARPGROUPS (NUM_WORKERS/kittens::WARPGROUP_WARPS)

#define ATTN_D_QK 128 // hardcoded into this kernel
#define ATTN_D_VO 64  // hardcoded into this kernel

#define WINDOW_WIDTH (64)

#define tile_q_smem   st_bf_4x4<wgmma_swizzle_l>
#define tile_k_smem   st_bf_4x4<wgmma_swizzle_l>
#define tile_qf_smem  st_bf_4x4<wgmma_swizzle_l>
#define tile_kf_smem  st_bf_4x4<wgmma_interleave_l>
#define tile_v_smem   st_bf_4x4<wgmma_interleave_l>
#define tile_o_smem   st_bf_4x4<swizzle_l>
#define tile_kv_smem  st_bf_4x4<wgmma_interleave_l>
#define tile_kv2_smem st_bf_4x4<wgmma_swizzle_l>

// struct reciprocal_op {
//     template<typename T> static __device__ inline T op(const T &x) { return T(1.f)/x; }
// };
// template<> __device__ inline bf16_2 reciprocal_op::op<bf16_2>(const bf16_2 &x) { return __float2bfloat162_rn(1.f)/x; }
// template<ducks::rt::all T>
// __device__ static inline void reciprocal(T &dst, const T &src) {
//     unary_map<reciprocal_op, T>(dst, src);
// }


template<ducks::rv::all RV, ducks::rt::row_layout RT>
__device__ static inline void diagonal(RV &dst, const RT &src) {
    static_assert(RT::height == 1 && RT::width == 1);
    static_assert(RV::inner_dim == 1);

    int row = (kittens::laneid() / 4);
    int col = (kittens::laneid() % 4) * 2;

    int srclane = row*4 + (row/2);

    using T = base_types::packing<typename RT::dtype>::unpacked_type;

    #pragma unroll
    for(int i = 0; i < 2; i++) {
        T source_val = row%2 == 0 ? src.tiles[0][0].data[i*3].x : src.tiles[0][0].data[i*3].y; // what to put out?
        T val = packed_shfl_sync(MASK_ALL, source_val, srclane);
        if (i == 0) {
            dst.data[0][0].x = val;
        } else {
            dst.data[0][0].y = val;
        }
    }    
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void hedgehog_attention(int n,
                        const CUtensorMap* tma_q,  const CUtensorMap* tma_k,
                        const CUtensorMap* tma_qf, const CUtensorMap* tma_kf,
                        const CUtensorMap* tma_v,
                        CUtensorMap* tma_o,
                        CUtensorMap* tma_kv) {

    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    tile_q_smem  (&q_smem)[2][2]  = al.allocate<tile_q_smem, 2, 2>(); // 16k * 2 (tictoc)
    tile_k_smem  (&k_smem)[2][2]  = al.allocate<tile_k_smem, 2, 2>(); // 16k * 2 (tictoc)
    tile_qf_smem (&qf_smem)[2][4] = al.allocate<tile_qf_smem, 2, 4>(); // 32k * 2 (tictoc)
    tile_kf_smem (&kf_smem)[2][4] = al.allocate<tile_kf_smem, 2, 4>(); // 32k * 2 (tictoc)
    tile_v_smem  (&v_smem)[2]     = al.allocate<tile_v_smem, 2>(); // 8k * 2 (tictoc)

    tile_kv_smem  (&kv_smem) = al.allocate<tile_kv_smem>(); // 8k
    tile_o_smem   (&o_smem)  = reinterpret_cast<tile_o_smem&>(kv_smem);
    tile_kv2_smem (&kv_smem_store)[4] = *reinterpret_cast<tile_kv2_smem(*)[4]>(q_smem[0]);

    // if(blockIdx.x == 0 && threadIdx.x == 0) printf("%llu\n", uint64_t(&kv_smem.data[0])-uint64_t(&q_smem[0][0].data[0]) + );

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int tic = 0, toc = 1;

    // launch tma loads
    __shared__ uint64_t qkv_barrier;

    int blocks = n / (64);

    // launch tma loads 
    if (warpid == 0) {
        tma::init_barrier(qkv_barrier, 1); // barrier for q and k
        tma::set_bytes(qkv_barrier, 6*size_bytes<tile_q_smem> + 6*size_bytes<tile_k_smem> + size_bytes<tile_v_smem>);

        int tile_idx = (blockIdx.x * blocks);
        #pragma unroll
        for(int i = 0; i < 2; i++) {
            tma::load_async(q_smem[tic][i], tma_q, qkv_barrier, tile_idx, i);
            tma::load_async(k_smem[tic][i], tma_k, qkv_barrier, tile_idx, i);
        }
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            tma::load_async(qf_smem[tic][i], tma_qf, qkv_barrier, tile_idx, i);
            tma::load_async(kf_smem[tic][i], tma_kf, qkv_barrier, tile_idx, i);
        }
        tma::load_async(v_smem[tic], tma_v, qkv_barrier, tile_idx);
    }

    rt_fl_1x4<> local_kv[4]; // 128 registers
    #pragma unroll
    for(int j = 0; j < 4; j++) {
        zero(local_kv[j]);
    }

    rt_fl_1x1<>::col_vec qk_diag; // for summing along the diagonal, for linear attention

    float last_max = -99999;
    rt_fl_1x1<>::col_vec max_vec;
    neg_infty(max_vec); // init low

    float last_norm = 0;
    rt_fl_1x1<>::col_vec norm_vec;
    zero(norm_vec);

    for (int block = 0; block < blocks; block++, tic^=1, toc^=1) {
        rt_fl_1x4 local_o; // 32 registers
        rt_fl_1x4 local_attn; // 32 registers
        rt_bf_1x4 local_attn_bf; // 16 registers

        tma::arrive_and_wait(qkv_barrier, tic);
        __syncthreads(); 

        if (warpid == 0) {
            tma::set_bytes(qkv_barrier, 6*size_bytes<tile_q_smem> + 6*size_bytes<tile_k_smem> + size_bytes<tile_v_smem>);
            if (block + 1 < blocks) {
                int tile_idx = (blockIdx.x * blocks) + (block + 1);
                #pragma unroll
                for(int i = 0; i < 2; i++) {
                    tma::load_async(q_smem[toc][i], tma_q, qkv_barrier, tile_idx, i);
                    tma::load_async(k_smem[toc][i], tma_k, qkv_barrier, tile_idx, i);
                }
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    tma::load_async(qf_smem[toc][i], tma_qf, qkv_barrier, tile_idx, i);
                    tma::load_async(kf_smem[toc][i], tma_kf, qkv_barrier, tile_idx, i);
                }
                tma::load_async(v_smem[toc], tma_v, qkv_barrier, tile_idx);
            }
        }

        __syncthreads();
        zero(local_attn);
        warpgroup::mma_fence(local_attn);
        #pragma unroll
        for(int j = 0; j < 2; j++) {
            warpgroup::mma_ABt(local_attn, q_smem[tic][j], k_smem[tic][j]);
            warpgroup::mma_commit_group(); 
        }
        warpgroup::mma_async_wait();

        __syncthreads();
        // now make causal
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            auto &attn_subtile = reinterpret_cast<rt_fl_1x1<>&>(local_attn.tiles[0][j]);
            if (j>warpid) zero(attn_subtile);
            else if (j==warpid) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
        }

        row_max(max_vec, local_attn, max_vec); // get row max
        // ensure also max'd with previous linear attn, across all rows
        max(max_vec, max_vec, last_max); // now done
        sub_row(local_attn, local_attn, max_vec); // subtract max off

        exp(local_attn, local_attn); // exp attn matrix for softmax

        // take exp of max_vec -- this will become normalization for linear term
        rt_fl_1x1<>::col_vec rescale_vec;
        sub(rescale_vec, max_vec, last_max); // we need to figure out how to adjust previous norm
        exp(rescale_vec, rescale_vec); // exp of max_vec
        copy(norm_vec, last_norm); // set all with linear attention norm
        div(norm_vec, norm_vec, rescale_vec); // normalize norm_vec individually according to the new max vec

        row_sum(norm_vec, local_attn, norm_vec); // we have now accumulated the norm with max accounted for

        copy(local_attn_bf, local_attn); // now stored in bf16

        // now we're going to reset the local_attn and use it to do the linear version, which we need to compute the normalizations constants.
        __syncthreads();
        zero(local_attn);
        warpgroup::mma_fence(local_attn);
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            warpgroup::mma_ABt(local_attn, qf_smem[tic][j], kf_smem[tic][j]);
            warpgroup::mma_commit_group(); 
        }
        warpgroup::mma_async_wait();
        // grab the diagonal from this, which we'll use to track the trace.
        diagonal<rt_fl_1x1<>::col_vec, rt_fl_1x1<>>(qk_diag, reinterpret_cast<rt_fl_1x1<>&>(local_attn.tiles[0][warpid])); // attn diagonal -- we will sum this onto the last_norm

        // basically, this is the linear attention component, tracking it for the next stage.
        float new_max;
        max(new_max, max_vec, last_max); // global softmax max from previous stages
        float rescale = __expf(last_max - new_max);
        last_norm *= rescale; // adjust previous trace using new max
        div(qk_diag, qk_diag, __expf(new_max)); // normalize diagonal to exp scaling, too
        sum(last_norm, qk_diag, last_norm); // add qk_diag to last_norm
        last_max = new_max;

        // if(block < 8) {
        //     if(blockIdx.x == 0 && threadIdx.x < 32) {
        //         printf("max_vec.x: %f, max_vec.y: %f\n", max_vec.data[0][0].x, max_vec.data[0][0].y);
        //     }
        //     if(blockIdx.x == 0 && threadIdx.x == 0) {
        //         printf("last_norm: %f, last_max: %f\n", last_norm, last_max);
        //     }
        // }

        zero(local_o);
        __syncthreads();

        for(int j = 0; j < 4; j++) {

            warpgroup::store(kv_smem, local_kv[j]);
            __syncthreads();

            warpgroup::mma_fence(local_o);
            warpgroup::mma_AB(local_o, qf_smem[tic][j], kv_smem);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            __syncthreads();

            warpgroup::mma_fence(local_kv[j]);
            warpgroup::mma_AtB(local_kv[j], kf_smem[tic][j], v_smem[tic]); // really AtB since k is transposed in advance
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            __syncthreads();
        }

        div_row(local_o, local_o, norm_vec); // normalize o according to exp max for numerical stability

        __syncthreads();
        warpgroup::mma_fence(local_o);
        warpgroup::mma_AB(local_o, local_attn_bf, v_smem[tic]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();

        __syncthreads(); 
        warpgroup::zero(kv_smem);
        __syncthreads(); 
        warpgroup::store(o_smem, local_o);
        __syncthreads(); 

        // launch tma store for o
        if (warpid == 0) {
            int tile_idx = (blockIdx.x * blocks) + block;
            tma::store_async(tma_o, o_smem, tile_idx);
            tma::store_commit_group(); 
        }
        __syncthreads();
        tma::store_async_wait();
        __syncthreads();
        warpgroup::zero(kv_smem);
        __syncthreads(); 
    }
    #pragma unroll
    for(int j = 0; j < 4; j++) {
        warpgroup::store(kv_smem_store[j], local_kv[j]);
        __syncthreads(); 
        if(warpid == 0) {
            tma::store_async(tma_kv, kv_smem_store[j], blockIdx.x*4 + j);
            tma::store_commit_group();
        }
        __syncthreads();
    }
    tma::store_async_wait();
}


#include "harness.impl"  // (comment out when using the code below)