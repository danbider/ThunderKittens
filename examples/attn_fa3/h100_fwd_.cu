// #define TORCH_COMPILE // defined by default for PyTorch bindings - to use cpp harness, comment this out

#ifdef TORCH_COMPILE
#include "src/kittens.cuh"
#else
#include "../../src/kittens.cuh"
#endif
#include <cuda/pipeline>

#include <cooperative_groups.h>
#include <cuda/barrier>

#define NUM_WORKERS (16)
#define NUM_WARPGROUPS (NUM_WORKERS/(kittens::WARPGROUP_WARPS))
#define NUM_WORKERS_KV (1)

#define NUM_PRODUCERS (1)
#define NUM_CONSUMERS (NUM_WARPGROUPS - NUM_PRODUCERS)

using namespace kittens;

using layout_q = kittens::ducks::st_layout::wgmma_swizzle; 
using layout_k = kittens::ducks::st_layout::wgmma_swizzle;
using layout_v = kittens::ducks::st_layout::wgmma_interleave;
using layout_o = kittens::ducks::st_layout::swizzle;

template<int D> struct fwd_attend_ker_tile_dims {
    constexpr static int tile_width = D/kittens::TILE_DIM;
    constexpr static int qo_height  = 4;
    constexpr static int kv_height  = 512/D;
};

template<int D>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker_dim(int N, const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, CUtensorMap* tma_o) {
    extern __shared__ int __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    constexpr int tile_width = fwd_attend_ker_tile_dims<D>::tile_width; // constants
    constexpr int qo_height  = fwd_attend_ker_tile_dims<D>::qo_height;
    constexpr int kv_height  = fwd_attend_ker_tile_dims<D>::kv_height;

    st_bf<qo_height, tile_width, layout_q> (&q_smem)   [NUM_CONSUMERS]  = al.allocate<st_bf<qo_height, tile_width, layout_q>,    NUM_CONSUMERS >();
    st_bf<kv_height, tile_width, layout_k> (&k_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_k>, 2, NUM_WORKERS_KV>();
    st_bf<kv_height, tile_width, layout_v> (&v_smem)[2][NUM_WORKERS_KV] = al.allocate<st_bf<kv_height, tile_width, layout_v>, 2, NUM_WORKERS_KV>();

    int warpid      = kittens::warpid();
    int warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    int kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows);

    using barrier = cuda::barrier<cuda::thread_scope_block>;

    int ready = 0, filled = 1;
    __shared__ barrier q_bar[2];     // [0: ready, 1: filled]
    __shared__ barrier k_bar[2][2]; // [NUM_STAGES][0: ready, 1: filled]
    __shared__ barrier v_bar[2][2]; // [NUM_STAGES][0: ready, 1: filled
    
    auto block = cooperative_groups::this_thread_block();
    if (threadIdx.x == 0) {
        init(q_bar[0], block.size()); 
        init(q_bar[1], block.size());

        for (int i = 0; i < 2; i++) {
            init(k_bar[i][0], block.size());
            init(k_bar[i][1], block.size());
            init(v_bar[i][0], block.size());
            init(v_bar[i][1], block.size());
        }
    }
    block.sync();

    static_assert(NUM_WORKERS == 12 || NUM_WORKERS == 16);
    if (warpgroupid < NUM_CONSUMERS) {
        // producer
        warpgroup::register_deallocate<NUM_WORKERS == 12 ? 24 : 32>();
        coalesced_group producer_wg = coalesced_threads();

        __shared__ uint64_t qsmem_barrier, ksmem_barrier, vsmem_barrier;

        int q_phasebit = 0;
        int k_phasebit = 0;
        int v_phasebit = 0;

        if (threadIdx.x == 0) {
            tma::init_barrier<st_bf<qo_height, tile_width, layout_q>, NUM_CONSUMERS >(qsmem_barrier, 1);
            tma::init_barrier<st_bf<kv_height, tile_width, layout_k>, NUM_WORKERS_KV>(ksmem_barrier, 1);
            tma::init_barrier<st_bf<kv_height, tile_width, layout_v>, NUM_WORKERS_KV>(vsmem_barrier, 1);

        }

        // check to see if q mem is open for business
        q_bar[ready].arrive_and_wait();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (int wg = 0; wg < NUM_CONSUMERS; wg++) { // load q
                int tile_idx = (blockIdx.y * NUM_CONSUMERS * gridDim.x) + (blockIdx.x * NUM_CONSUMERS) + wg;
                tma::load_async((q_smem[wg]), tma_q, qsmem_barrier, tile_idx); 
            }
        }

        producer_wg.sync();
        
        tma::arrive_and_wait(qsmem_barrier, q_phasebit);
        q_phasebit ^= 1;

        // notify consumers that q is ready
        barrier::arrival_token token = q_bar[filled].arrive();
        // we are forever done with Q

        k_bar[0][ready].arrive_and_wait();
        v_bar[0][ready].arrive_and_wait();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (int w = 0; w < NUM_WORKERS_KV; w++) {
                int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + (0 * NUM_WORKERS_KV) + w; 
                tma::load_async((k_smem[0][w]), tma_k, ksmem_barrier, tile_idx);
                tma::load_async((v_smem[0][w]), tma_v, vsmem_barrier, tile_idx);
            }
        }
        producer_wg.sync();

        for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
            tma::arrive_and_wait(ksmem_barrier, k_phasebit);
            k_phasebit ^= 1;
            k_bar[kv_idx % 2][filled].arrive();

            tma::arrive_and_wait(vsmem_barrier, v_phasebit);
            v_phasebit ^= 1;
            v_bar[kv_idx % 2][filled].arrive();

            if (kv_idx + 1 < kv_blocks) {
                k_bar[(kv_idx + 1) % 2][ready].arrive_and_wait();
                v_bar[(kv_idx + 1) % 2][ready].arrive_and_wait();

                if (warpid % kittens::WARPGROUP_WARPS == 0) {
                    tma::set_bytes(ksmem_barrier, NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));
                    tma::set_bytes(vsmem_barrier, NUM_WORKERS_KV * v_smem[0][0].num_elements * sizeof(bf16));

                    for (int w = 0; w < NUM_WORKERS_KV; w++) {
                        int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                        tma::load_async((k_smem[(kv_idx + 1) % 2][w]), tma_k, ksmem_barrier, tile_idx);
                        tma::load_async((v_smem[(kv_idx + 1) % 2][w]), tma_v, vsmem_barrier, tile_idx);
                    }
                }
            }
            producer_wg.sync();
        }
    }
    else {
        // consumer
        warpgroup::register_allocate<NUM_WORKERS == 12 ? 240 : 160>();

        warpgroupid -= NUM_CONSUMERS;

        // all memory is open for business
        q_bar[ready].arrive();
        for (int i = 0; i < 2; i++) {
            k_bar[i][ready].arrive();
            v_bar[i][ready].arrive();
        }

        rt_fl<1, kv_height> att_block;
        rt_bf<1, kv_height> att_block_mma;
        rt_fl<1, tile_width> o_prev;
        col_vec<rt_fl<1, kv_height>> max_vec_last, max_vec;
        col_vec<rt_fl<1, kv_height>> norm_vec_last, norm_vec;

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_prev);

        // first thing needed is the Q chunk
        q_bar[filled].arrive_and_wait();

        if constexpr (D == 64) { 
            warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); 
        } 
        else { 
            warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); 
        }

        // wait for the kv_idx == 0 k chunk
        k_bar[0 % 2][filled].arrive_and_wait();

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[0 % 2][0]);
        warpgroup::mma_commit_group();

        copy(norm_vec_last, norm_vec);
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait();

        k_bar[0 % 2][ready].arrive();

        row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
        sub_row(att_block, att_block, max_vec);
        exp(att_block, att_block);

        sub(max_vec_last, max_vec_last, max_vec);
        exp(max_vec_last, max_vec_last);
        mul(norm_vec, norm_vec, max_vec_last);

        row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
        div_row(att_block, att_block, norm_vec);

        mul(norm_vec_last, norm_vec_last, max_vec_last);
        div(norm_vec_last, norm_vec_last, norm_vec);

        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

        for (int kv_idx = 1; kv_idx < kv_blocks; kv_idx++) {
            k_bar[kv_idx % 2][filled].arrive_and_wait();

            warpgroup::mma_fence(att_block);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[kv_idx % 2][0]);
            warpgroup::mma_commit_group();

            v_bar[(kv_idx - 1) % 2][filled].arrive_and_wait();

            warpgroup::mma_fence(o_prev);
            warpgroup::mma_AB(o_prev, att_block_mma, v_smem[(kv_idx - 1) % 2][0]);
            warpgroup::mma_commit_group();

            warpgroup::mma_async_wait<1>();
        }

    }



    neg_infty(max_vec); // zero registers for the Q chunk
    zero(norm_vec);
    zero(o_prev);
    __syncthreads();

    tma::arrive_and_wait(qsmem_barrier, q_phasebit);
    q_phasebit ^= 1;

    if constexpr (D == 64) { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.125f)); } 
    else { warpgroup::mul(q_smem[warpgroupid], q_smem[warpgroupid], __float2bfloat16(0.08838834764f)); }

    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic ^= 1, toc ^= 1) {
        tma::arrive_and_wait(kvsmem_barrier, kv_phasebit);
        kv_phasebit ^= 1;

        __syncthreads();
        if (warpid == 0) {
            tma::set_bytes(kvsmem_barrier, 2 * NUM_WORKERS_KV * k_smem[0][0].num_elements * sizeof(bf16));

            if (kv_idx + 1 < kv_blocks) {
                for (int w = 0; w < NUM_WORKERS_KV; w++) {        
                    int tile_idx = (blockIdx.y * NUM_WORKERS_KV * kv_blocks) + ((kv_idx + 1) * NUM_WORKERS_KV) + w; 
                    tma::load_async((k_smem[toc][w]), tma_k, kvsmem_barrier, tile_idx); 
                    tma::load_async((v_smem[toc][w]), tma_v, kvsmem_barrier, tile_idx);
                }
            }
        }

        warpgroup::mma_fence(att_block);
        warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[tic][0]);
        warpgroup::mma_commit_group();

        copy(norm_vec_last, norm_vec);
        copy(max_vec_last,  max_vec);

        warpgroup::mma_async_wait();

        row_max(max_vec, att_block, max_vec); // accumulate onto the max_vec
        sub_row(att_block, att_block, max_vec);
        exp(att_block, att_block);

        sub(max_vec_last, max_vec_last, max_vec);
        exp(max_vec_last, max_vec_last);
        mul(norm_vec, norm_vec, max_vec_last);

        row_sum(norm_vec, att_block, norm_vec); // accumulate onto the norm_vec
        div_row(att_block, att_block, norm_vec);

        mul(norm_vec_last, norm_vec_last, max_vec_last);
        div(norm_vec_last, norm_vec_last, norm_vec);

        copy(att_block_mma, att_block); // convert to bf16 for mma
        mul_row(o_prev, o_prev, norm_vec_last); // normalize o_prev in advance of mma'ing onto it

        warpgroup::mma_fence(o_prev);
        warpgroup::mma_AB(o_prev, att_block_mma, v_smem[tic][0]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();
    }

    auto (*o_smem) = reinterpret_cast<st_bf<qo_height, tile_width, layout_o>(*)>(q_smem); // reuse q memory
    warpgroup::store(o_smem[warpgroupid], o_prev); 
    __syncthreads();
    
    if (warpid % 4 == 0) { // store o
        int tile_idx = (blockIdx.y * NUM_WARPGROUPS * gridDim.x) + (blockIdx.x * NUM_WARPGROUPS) + warpgroupid;
        tma::store_async(tma_o, (o_smem[warpgroupid]), tile_idx); 
        tma::store_commit_group(); 
    }

    tma::store_async_wait();
}

#ifdef TORCH_COMPILE
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>

void attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {

    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);

    auto batch   = q.size(0);
    auto heads   = q.size(1);
    auto N       = q.size(2);
    auto D       = q.size(3);

    auto threads = NUM_WORKERS * kittens::WARP_THREADS;

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "k must be bf16");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "v must be bf16");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "o must be bf16");

    // make sure sequence length is multiple of 128 for now
    TORCH_CHECK(N % (NUM_WORKERS * kittens::TILE_DIM) == 0, "Please pad sequence length to be multiple of 128");

    // make sure D = 64 or 128
    TORCH_CHECK(D == 64 || D == 128, "Currently, only D = 64 or 128 is supported");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
    bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

    if (D == 64) {

        CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));
        CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
        CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::kv_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::kv_height * 16));
        CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<64>::qo_height, fwd_attend_ker_tile_dims<64>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<64>::qo_height * 16));

        unsigned long mem_size = 112000;
        cudaFuncSetAttribute(fwd_attend_ker_dim<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

        fwd_attend_ker_dim<64><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
    }
    else {
        CUtensorMap* tma_q_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_q>>(q_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));
        CUtensorMap* tma_k_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_k>>(k_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
        CUtensorMap* tma_v_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_v>>(v_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::kv_height * 16));
        CUtensorMap* tma_o_d = tma::allocate_and_create_tensor_map<kittens::st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width, layout_o>>(o_bf, (batch*heads*N)/(fwd_attend_ker_tile_dims<128>::qo_height * 16));

        unsigned long mem_size = 112000;
        cudaFuncSetAttribute(fwd_attend_ker_dim<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        dim3 grid(N/(NUM_WORKERS*kittens::TILE_DIM), batch*heads, 1);

        fwd_attend_ker_dim<128><<<grid, threads, mem_size>>>(N, tma_q_d, tma_k_d, tma_v_d, tma_o_d);
    }
    
    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness_h100_fwd_.impl"
#endif