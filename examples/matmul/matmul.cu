#include "../../src/kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>

using namespace kittens;

#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 32
#define NUM_WARPS 4
#define GROUP_SIZE 8

__global__ void matmul(bf16 *__a__, bf16 *__b__, bf16 *__c__, int N)
{ // N must be a multiple of 128!!

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // int pid = blockIdx.x + blockIdx.y * gridDim.x;

    // auto num_pid_m = N / 128;
    // auto num_pid_n = N / 128;
    // auto num_pid_in_group = GROUP_SIZE * num_pid_n;
    // auto group_id = pid / num_pid_in_group;
    // auto first_pid_m = group_id * GROUP_SIZE;
    // auto group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE);
    // auto pid_m = first_pid_m + (pid % group_size_m);
    // auto pid_n = (pid % num_pid_in_group) / group_size_m;

    // auto block_row = pid_m;
    // auto block_col = pid_n;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al = shared_allocator::create_allocator((int *)&__shm[0]);

    typedef st_bf<8, 2, ducks::st_layout::xor_swizzle> st_ab;

    st_ab(&st_a)[2] = al.allocate<st_ab, 2>();
    st_ab(&st_b)[2] = al.allocate<st_ab, 2>();

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> bar;
    if (threadIdx.x == 0)
    {
        init(&bar, block.size());
    }
    __syncthreads();

    // 32 x 16 A
    rt_bf<2, 1, ducks::rt_layout::row> a_reg;

    // 128 x 16 B
    rt_bf<8, 1, ducks::rt_layout::row> b_reg;

    // 32 x 128 output
    rt_fl<2, 8, ducks::rt_layout::row> c_reg;

    int num_warps = blockDim.x / 32;

    auto rows_per_warp = BLOCK_SIZE_N / NUM_WARPS;

    zero(c_reg); // reset

    int tic = 0;
    int toc = 1;

    auto a = __a__ + block_row * BLOCK_SIZE_N * N;
    auto b = __b__ + block_col * BLOCK_SIZE_N * N;
    auto c = __c__ + block_row * BLOCK_SIZE_N * N + block_col * BLOCK_SIZE_N;

    // subtile_inplace<2, 1>(st_a[tic], warpid(), k);
    auto subtile_a = subtile_inplace<2, 2>(st_a[tic], warpid(), 0);
    auto subtile_b = subtile_inplace<2, 2>(st_b[tic], warpid(), 0);

    // auto subtile_a = st_a[tic].template subtile<2, 2>(warpid(), 0);
    // auto subtile_b = st_b[tic].template subtile<2, 2>(warpid(), 0);

    auto ab_offset = rows_per_warp * warpid() * N;

    load_async(subtile_a, a + ab_offset, N, bar);
    load_async(subtile_b, b + ab_offset, N, bar);

    // now iterate through A, B
    const int accum_blocks = N / BLOCK_SIZE_K;

    for (int i = 0; i < accum_blocks; i++)
    {
        bar.arrive_and_wait();
        int next = i + 1;

        if (next < accum_blocks)
        {

            // auto subtile_a = st_a[toc].template subtile<2, 2>(warpid(), 0);
            // auto subtile_b = st_b[toc].template subtile<2, 2>(warpid(), 0);

            auto subtile_a = subtile_inplace<2, 2>(st_a[toc], warpid(), 0);
            auto subtile_b = subtile_inplace<2, 2>(st_b[toc], warpid(), 0);

            auto ab_offset = rows_per_warp * warpid() * N + next * BLOCK_SIZE_K;

            load_async(subtile_a, a + ab_offset, N, bar);
            load_async(subtile_b, b + ab_offset, N, bar);
        }

#pragma unroll // actually important
        for (int k = 0; k < 2; k++)
        {
            // auto comp_subtile_a = st_a[tic].template subtile<2, 1>(warpid(), k);
            // auto comp_subtile_b = st_b[tic].template subtile<8, 1>(0, k);

            auto comp_subtile_a = subtile_inplace<2, 1>(st_a[tic], warpid(), k);
            auto comp_subtile_b = subtile_inplace<8, 1>(st_b[tic], 0, k);

            load(a_reg, comp_subtile_a);
            load(b_reg, comp_subtile_b);

            dot(c_reg, a_reg, b_reg, c_reg);
        }

        tic ^= 1;
        toc ^= 1;
    }

    auto c_offset = rows_per_warp * warpid() * N;
    store(c + c_offset, c_reg, N);
}

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
inline void __cudaCheckError(const char *file, const int line)
{
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#define MATMUL_N 2048

int main(int argc, char **argv)
{
    // TODO: consider doing sequential kernel launches to force batches dimension element to execute sequentially,
    // which may increase the probability of L2 cache hits on KV

    std::cout << "Entered main!" << std::endl;

    // create dummy variables that are the right size
    constexpr int TOTAL_ELEMENTS = MATMUL_N * MATMUL_N;

    float *a = new float[TOTAL_ELEMENTS];
    float *b = new float[TOTAL_ELEMENTS];
    float *c = new float[TOTAL_ELEMENTS];
    float *o = new float[TOTAL_ELEMENTS];
    float *o_ref = new float[TOTAL_ELEMENTS];

    bf16 *a_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *b_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *c_bf = new bf16[TOTAL_ELEMENTS];
    bf16 *o_bf = new bf16[TOTAL_ELEMENTS];

    if (argc > 1)
    {
        std::ifstream infile(argv[1]);

        std::cout << "Starting to enter!" << std::endl;

        for (int i = 0; i < TOTAL_ELEMENTS; i++)
            infile >> a[i];
        std::cout << "Finished loading A" << std::endl;
        for (int i = 0; i < TOTAL_ELEMENTS; i++)
            infile >> b[i];
        std::cout << "Finished loading B" << std::endl;
        for (int i = 0; i < TOTAL_ELEMENTS; i++)
            infile >> c[i];
        std::cout << "Finished loading C" << std::endl;
        for (int i = 0; i < TOTAL_ELEMENTS; i++)
            infile >> o_ref[i];
        std::cout << "Finished loading O" << std::endl;

        std::cout << "Finished loading file from " << argv[1] << "!" << std::endl;
    }
    else
    {
        std::cout << "WARNING: No file provided, correctness is meaningless" << std::endl;
    }

    // replicate into heads
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
    {
        a_bf[i] = __float2bfloat16(a[i]);
        b_bf[i] = __float2bfloat16(b[i]);
        c_bf[i] = __float2bfloat16(c[i]);
    }

    bf16 *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_b, TOTAL_ELEMENTS * sizeof(bf16));
    cudaMalloc(&d_c, TOTAL_ELEMENTS * sizeof(bf16));

    cudaMemcpy(d_a, a_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c_bf, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyHostToDevice);

    // tile size * bfloat * a and b * tic and toc
    unsigned long mem_size = 128 * 32 * 2 * 2 * 2;

    dim3 grid(MATMUL_N / BLOCK_SIZE_N, MATMUL_N / BLOCK_SIZE_N);

    constexpr bool run_once_and_finish = true;

    if (run_once_and_finish)
    {
        matmul<<<grid, BLOCK_SIZE_N, mem_size>>>(d_a, d_b, d_c, MATMUL_N);
        cudaDeviceSynchronize();
        return 0;
    }


    constexpr int warmup_milis = 1000;
    constexpr int milis = 1000;
    constexpr int est_iters = 10;

    cudaDeviceSynchronize();

    const auto est_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < est_iters; i++)
    {
        matmul<<<grid, BLOCK_SIZE_N, mem_size>>>(d_a, d_b, d_c, MATMUL_N);
    }
    cudaDeviceSynchronize();

    const auto est_finish = std::chrono::high_resolution_clock::now();
    auto est_time = std::chrono::duration_cast<std::chrono::microseconds>(est_finish - est_start).count() / est_iters;

    auto warmup_iters = warmup_milis * 1000 / est_time;
    auto ITER = milis * 1000 / est_time;

    std::cout << "Warmup iters: " << warmup_iters << std::endl;
    std::cout << "Iters: " << ITER << std::endl;

    for (int i = 0; i < warmup_iters; i++)
    {
        matmul<<<grid, BLOCK_SIZE_N, mem_size>>>(d_a, d_b, d_c, MATMUL_N);
    }

    cudaDeviceSynchronize();

    std::cout << "Starting kernel\n";
    const auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < ITER; j++)
    {
        matmul<<<grid, BLOCK_SIZE_N, mem_size>>>(d_a, d_b, d_c, MATMUL_N);
    }
    cudaDeviceSynchronize();
    std::cout << "Finished kernel\n";

    const auto finish = std::chrono::high_resolution_clock::now();
    CudaCheckError();

    auto avg_time = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() / ITER;
    std::cout << "Average execution time: " << avg_time << " us" << std::endl;
    auto tflops = (2.0 * MATMUL_N * MATMUL_N * MATMUL_N) / (avg_time / 1e6) / 1e12;
    std::cout << "TFLOPS: " << tflops << std::endl;
    auto theoretical_flops = 165.2;
    std::cout << "Util: " << (tflops / theoretical_flops) * 100 << "%" << std::endl;

    // check correctness
    cudaMemcpy(o_bf, d_c, TOTAL_ELEMENTS * sizeof(bf16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
    {
        o[i] = __bfloat162float(o_bf[i]);
    }

    bool good = true;
    bool should_write = false;
    std::ofstream o_ref_file;
    std::ofstream o_file;
    std::ofstream diff_file;
    if (should_write)
    {
        o_ref_file.open("printouts/o_ref.txt");
        o_file.open("printouts/o.txt");
        diff_file.open("printouts/diff.txt");
    }
    should_write &= o_ref_file.is_open();
    for (int i = 0; i < TOTAL_ELEMENTS; i++)
    {
        float diff = o[i] - o_ref[i];
        if (should_write)
        {
            o_ref_file << o_ref[i] << ' ';
            o_file << o[i] << ' ';
            diff_file << diff << ' ';
        }
        if (abs(diff) > 0.01 || isnan(diff))
        {
            good = false;
        }
    }
    if (good)
        std::cout << "Correct :)\n";
    else
        std::cout << "Incorrect :(\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a, b, c, o, o_ref;
    delete[] a_bf, b_bf, c_bf;

    return 0;
}