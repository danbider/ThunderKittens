
#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

using namespace kittens;

// pointwise-gemm
#define a_tile st_bf_4x4
#define b_tile st_bf_4x4
#define c_tile st_bf_4x4

template<int _NUM_CONSUMER_WARPGROUPS>
struct producer_consumer_parameters { 
    static constexpr int NUM_CONSUMER_WARPGROUPS = _NUM_CONSUMER_WARPGROUPS;
    static_assert(NUM_CONSUMER_WARPGROUPS >= 2 && NUM_CONSUMER_WARPGROUPS <= 6); // The register alloc is only set up for this range.
    static constexpr int NUM_CONSUMER_WARPS      = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_WARPS               = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // producers, too
    static constexpr int NUM_THREADS             = NUM_WARPS * WARP_THREADS;
    static constexpr int NUM_PRODUCER_REG        = NUM_CONSUMER_WARPGROUPS == 2 ? 32 : 24;
    static constexpr int NUM_CONSUMER_REG        = 480/NUM_CONSUMER_WARPGROUPS-8; // valid up to 6 consumer warpgroups
};

struct globals {
    int n_blocks;
    const CUtensorMap* A_tma;
    const CUtensorMap* B_tma;
    CUtensorMap* C_tma;
    __host__ __device__ inline globals(int n_blocks, const CUtensorMap* A_tma, const CUtensorMap* B_tma, CUtensorMap* C_tma) :
        n_blocks(n_blocks), A_tma(A_tma), B_tma(B_tma), C_tma(C_tma) {}
};

template<int _NUM_CONSUMER_WARPGROUPS>
struct block { 
    // the chunk of data that the producer and consumer are working on
    a_tile (&a_block)[_NUM_CONSUMER_WARPGROUPS];
    b_tile (&b_block);

    // this does a n inidialization directly, rather than assigning to the body of the constructor
    __device__ inline block(a_tile (&a_block)[_NUM_CONSUMER_WARPGROUPS], b_tile (&b_block)) : a_block(a_block), b_block(b_block) {}
};

struct producer_consumer {
    static constexpr int NUM_CONSUMER_WARPGROUPS = 2;
    using params = producer_consumer_parameters<NUM_CONSUMER_WARPGROUPS>;
    using block = block<NUM_CONSUMER_WARPGROUPS>;

    struct producer {
        struct state {
            int row_idx, col_idx; // persistent registers
        };
        __device__ static void setup(state &s, globals &g) { // setup and load the first iteration
            warpgroup::decrease_registers<params::NUM_PRODUCER_REG>(); // decrease registers for the producer warpgroup
            s.row_idx = blockIdx.x * NUM_CONSUMER_WARPGROUPS; // tiles vertical per block
            s.col_idx = blockIdx.y; // just 1 tile horizontal per block
        }
        __device__ static void load(state &s, block &b, globals &g, kittens::barrier &bar, int iter) { 
            // barrier for the producer to load into
            if(warpgroup::warpid() == 0) {
                tma::expect_bytes(bar, size_bytes<a_tile>*NUM_CONSUMER_WARPGROUPS + size_bytes<b_tile>);
                #pragma unroll
                for(int i = 0; i < NUM_CONSUMER_WARPGROUPS; i++) {
                    tma::load_async(b.a_block[i], g.A_tma, bar, s.row_idx+i, iter);
                }
                tma::load_async(b.b_block, g.B_tma, bar, iter, s.col_idx);
            }
            // apply pointwise nonlinearity to A
            
        }
        __device__ static void finish(state &s, globals &g) {}
    };

    struct consumer {
        struct state {
            rt_fl<1,c_tile::width> acc;
            c_tile &out_block;
            __host__ __device__ inline state(c_tile &out_block) : out_block(out_block) {}
        }; // persistent registers; none needed for this kernel.
        __device__ static void setup(state &s, globals &g) { // setup locals for before the first iteration
            warpgroup::increase_registers<params::NUM_CONSUMER_REG>();
            zero(s.acc);
        }
        __device__ static void compute(state &s, block &b, globals &g, int iter) {
            warpgroup::mma_fence(s.acc);
            warpgroup::mma_AB(s.acc, b.a_block[warpgroup::groupid()], b.b_block);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
        }
        __device__ static void finish(state &s, globals &g) {
            warpgroup::store(s.out_block, s.acc);
            warpgroup::sync(); // writes to shared memory are now visible
            if(warpgroup::warpid() == 0) { // first warp stores
                tma::store_async(g.C_tma, s.out_block, blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), blockIdx.y);
                tma::store_commit_group();
            }
            tma::store_async_read_wait(); // this isn't really necessary, but it illustrates the principle.
            warpgroup::sync();
        }
    };
};

constexpr int PIPE_STAGES = 4;
__device__ inline int advance(int ring) { return (ring + 1) % PIPE_STAGES; }
__device__ inline int retreat(int ring) { return (ring + PIPE_STAGES-1) % PIPE_STAGES; }

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
__global__ __launch_bounds__(producer_consumer::params::NUM_THREADS, 1)
void gpu_gemm(globals g) {
    using pc = producer_consumer;

    extern __shared__ int __shm2[];
    shared_allocator alloc(&__shm2[0]); // allocate shared memory
    a_tile (&a_smem) [PIPE_STAGES][producer_consumer::params::NUM_CONSUMER_WARPGROUPS] = alloc.allocate<a_tile, PIPE_STAGES, producer_consumer::params::NUM_CONSUMER_WARPGROUPS>();
    b_tile (&b_smem) [PIPE_STAGES]                                                     = alloc.allocate<b_tile, PIPE_STAGES>();
    c_tile (&c_smem) [producer_consumer::params::NUM_CONSUMER_WARPGROUPS]              = reinterpret_cast<c_tile(&)[producer_consumer::params::NUM_CONSUMER_WARPGROUPS]>(a_smem); // ovewrwrite at the end
    block<producer_consumer::params::NUM_CONSUMER_WARPGROUPS> blocks[] = {
        block(a_smem[0], b_smem[0]),
        block(a_smem[1], b_smem[1]),
        block(a_smem[2], b_smem[2]),
        block(a_smem[3], b_smem[3])
    };

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier producer_arrived[PIPE_STAGES], consumer_arrived[PIPE_STAGES];
    int ring = 0; // these are used to track the two-stage pipeline.
    if (warpid() < PIPE_STAGES) { // a single warp (in fact a single thread) does these.
        init_barrier(producer_arrived[warpid()], 0, 1); // needs to wait on just one memory transaction, each
        init_barrier(consumer_arrived[warpid()], pc::params::NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroup::groupid() == pc::params::NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        typename pc::producer::state s;
        pc::producer::setup(s, g);
        pc::producer::load(s, blocks[ring], g, producer_arrived[ring], 0); // load initial block
        if constexpr (PIPE_STAGES>2) pc::producer::load(s, blocks[advance(ring)], g, producer_arrived[advance(ring)], 1); // load second block for pipeline
        if constexpr (PIPE_STAGES>3) pc::producer::load(s, blocks[advance(advance(ring))], g, producer_arrived[advance(advance(ring))], 2); // load third block for pipeline
        for (int block_idx = PIPE_STAGES-1; block_idx < g.n_blocks; block_idx++, ring=advance(ring)) {
            int ring_load = retreat(ring); // maximally advanced, pipe_stages-1 times
            pc::producer::load(s, blocks[ring_load], g, producer_arrived[ring_load], block_idx);
            wait(consumer_arrived[ring], ((block_idx-(PIPE_STAGES-1))/PIPE_STAGES)%2); // phase changes at half the rate of the tic/toc
        }
        pc::producer::finish(s, g);
    }
    else { // other warpgroups are consumers
        typename pc::consumer::state s(c_smem[warpgroup::groupid()]);
        pc::consumer::setup(s, g);
        // Option 1: simple PC
        // for (int block_idx = 0; block_idx < g.n_blocks; block_idx++, ring=advance(ring)) {
        //     wait(producer_arrived[ring], (block_idx/PIPE_STAGES)%2); // wait for memory to arrive
        //     pc::consumer::compute(s, blocks[ring], g, block_idx);
        //     if(laneid() == 0) arrive(consumer_arrived[ring]); // overlap arrival for previous with this matmul
        // }
        // Option 2: hide barrier stuff during the wgmma's, which gives another ~20 TFLOPs
        wait(producer_arrived[ring], 0); // wait for initial memory to arrive
        silu(
            blocks[ring].a_block[warpgroup::groupid()], 
            blocks[ring].a_block[warpgroup::groupid()]
        ); 
        warpgroup::mma_fence(s.acc);
        warpgroup::mma_AB(s.acc, blocks[ring].a_block[warpgroup::groupid()], blocks[ring].b_block); // launch first one, don't wait.
        warpgroup::mma_commit_group();
        ring = advance(ring);
        for (int block_idx = 1; block_idx < g.n_blocks; block_idx++, ring=advance(ring)) {
            wait(producer_arrived[ring], (block_idx/PIPE_STAGES)%2); // wait for next memory to arrive while we wait for tensor cores
            silu(
                blocks[ring].a_block[warpgroup::groupid()], 
                blocks[ring].a_block[warpgroup::groupid()]
            ); 
            warpgroup::mma_async_wait(); // previous is finished
            warpgroup::mma_fence(s.acc);
            warpgroup::mma_AB(s.acc, blocks[ring].a_block[warpgroup::groupid()], blocks[ring].b_block);
            warpgroup::mma_commit_group();
            if(laneid() == 0) arrive(consumer_arrived[retreat(ring)]); // overlap arrival for previous with this matmul
        }
        warpgroup::mma_async_wait();
        if(laneid() == 0) arrive(consumer_arrived[retreat(ring)]); // final one finished
        // Common writeout
        pc::consumer::finish(s, g);
    }
}

#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void pointwise_gemm(
    torch::Tensor A, torch::Tensor B, torch::Tensor C
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // add some shape assertions 
    bool inner_dim     = A.size(1) == B.size(0); // check the A, B inner dimension
    bool out_dim_rows  = A.size(0) == C.size(0); // check the output shape
    bool out_dim_cols  = B.size(1) == C.size(1); // check the output shape

    TORCH_CHECK(inner_dim, "A and B must have the same inner dimension");
    TORCH_CHECK(out_dim_rows, "A and C must share dim");
    TORCH_CHECK(out_dim_cols, "B and C must share dim");
    TORCH_CHECK(A.scalar_type() == c10::ScalarType::BFloat16, "A must be bf16");
    TORCH_CHECK(B.scalar_type() == c10::ScalarType::BFloat16, "B must be bf16");
    TORCH_CHECK(C.scalar_type() == c10::ScalarType::BFloat16, "C must be bf16");

    // convert to bf16
    c10::BFloat16* A_ptr = A.data_ptr<c10::BFloat16>();
    c10::BFloat16* B_ptr = B.data_ptr<c10::BFloat16>();
    c10::BFloat16* C_ptr = C.data_ptr<c10::BFloat16>();

    const bf16* d_A  = reinterpret_cast<const bf16*>(A_ptr);
    const bf16* d_B  = reinterpret_cast<const bf16*>(B_ptr);
          bf16* d_C  = reinterpret_cast<bf16*>(C_ptr);

    CUtensorMap* tma_A_d = tma::allocate_and_create_tensor_map<a_tile>(d_A, M/a_tile::rows, K/a_tile::cols);
    CUtensorMap* tma_B_d = tma::allocate_and_create_tensor_map<b_tile>(d_B, K/b_tile::rows, N/b_tile::cols);
    CUtensorMap* tma_C_d = tma::allocate_and_create_tensor_map<c_tile>(d_C, M/c_tile::rows, N/c_tile::cols);

    unsigned long mem_size = 200000; // need to launch two blocks if possible.

    cudaFuncSetAttribute(
        gpu_gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    ); 

    dim3 grid(M / (c_tile::rows*producer_consumer::params::NUM_CONSUMER_WARPGROUPS), N / c_tile::cols); // rows, cols
    dim3 block(producer_consumer::params::NUM_THREADS);
    gpu_gemm<<<grid, block, mem_size>>>(globals(K/a_tile::cols, tma_A_d, tma_B_d, tma_C_d));


    CHECK_CUDA_ERROR(cudaGetLastError());
}

