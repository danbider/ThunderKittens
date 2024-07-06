# ThunderKittens Memory Operations Documentation

This document provides detailed information about the memory operations in ThunderKittens, including examples for both vector and tile operations where applicable.

## Table of Contents
1. [Global to Register](#global-to-register)
2. [Global to Shared](#global-to-shared)
3. [Shared to Register](#shared-to-register)
4. [Tensor Map Operations (TMA)](#tensor-map-operations-tma)
5. [Distributed Shared Memory (DSMEM) Operations](#distributed-shared-memory-dsmem-operations)

## Global to Register

### load

Loads data from global memory into a register tile or vector.

#### Tile Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    kittens::rt_fl_2x1<> my_tile;
    kittens::load(my_tile, global_data, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    kittens::row_vec<kittens::rt_fl_2x4<>> my_vector;
    kittens::load(my_vector, global_data);
}
```

### store

Stores data from a register tile or vector to global memory.

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    kittens::rt_fl_2x1<> my_tile;
    // ... operations on my_tile ...
    kittens::store(global_data, my_tile, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    kittens::row_vec<kittens::rt_fl_2x4<>> my_vector;
    // ... operations on my_vector ...
    kittens::store(global_data, my_vector);
}
```

## Global to Shared

### load

Loads data from global memory into a shared memory tile or vector. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    kittens::load(shared_tile, global_data, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    __shared__ kittens::sv_bf_2 shared_vector;
    kittens::load(shared_vector, global_data);
}
```

### store

Stores data from a shared memory tile or vector to global memory. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    // ... operations on shared_tile ...
    kittens::store(global_data, shared_tile, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    __shared__ kittens::sv_bf_2 shared_vector;
    // ... operations on shared_vector ...
    kittens::store(global_data, shared_vector);
}
```

### load_async (non-TMA)

Asynchronously loads data from global memory into a shared memory tile or vector without using Tensor Memory Accelerator (TMA).

#### Tile Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    kittens::load_async(shared_tile, global_data, row_stride, barrier);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    __shared__ kittens::sv_bf_2 shared_vector;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    kittens::load_async(shared_vector, global_data, barrier);
}
```

### store_async (non-TMA)

Asynchronously stores data from a shared memory tile or vector to global memory without using Tensor Memory Accelerator (TMA).

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    // ... operations on shared_tile ...
    kittens::store_async(global_data, shared_tile, row_stride, barrier);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    __shared__ kittens::sv_bf_2 shared_vector;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    // ... operations on shared_vector ...
    kittens::store_async(global_data, shared_vector, barrier);
}
```

## Shared to Register

### load

Loads data from a shared memory tile or vector into a register tile or vector. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel() {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    rt_bf_2x1<> reg_tile;
    kittens::load(reg_tile, shared_tile);
}
```

#### Vector Example:

```cuda
__global__ void kernel() {
    __shared__ kittens::sv_bf_2 shared_vector;
    kittens::row_vec<kittens::rt_bf_2x2<>> reg_vector;
    kittens::load(reg_vector, shared_vector);
}
```

### store

Stores data from a register tile or vector to a shared memory tile or vector. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel() {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    rt_fl_2x1<> reg_tile;
    // ... operations on reg_tile ...
    kittens::store(shared_tile, reg_tile);
}
```

#### Vector Example:

```cuda
__global__ void kernel() {
    __shared__ kittens::sv_bf_2 shared_vector;
    kittens::row_vec<kittens::rt_bf_2x2<>> reg_vector;
    // ... operations on reg_vector ...
    kittens::store(shared_vector, reg_vector);
}
```

## Tensor Map Operations (TMA)

### create_tensor_map

Creates a tensor map descriptor for use with TMA operations.

```cuda
__host__ void host_function(const kittens::bf16* global_data, int blocks_height, int blocks_width) {
    CUtensorMap tma_descriptor;
    kittens::tma::create_tensor_map<kittens::st_bf_2x1<>>(&tma_descriptor, global_data, blocks_height, blocks_width);
}
```

### allocate_and_create_tensor_map

Allocates memory for and creates a tensor map descriptor for use with TMA operations.

```cuda
__host__ void host_function(const kittens::bf16* global_data, int blocks_height, int blocks_width) {
    CUtensorMap* tma_descriptor = kittens::tma::allocate_and_create_tensor_map<kittens::st_bf_2x1<>>(global_data, blocks_height, blocks_width);
}
```

### prefetch

Prefetches data from global memory into L2 cache using TMA.

#### Tile Example:

```cuda
__global__ void kernel(const CUtensorMap* src_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::prefetch(shared_tile, src_tma_map, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const CUtensorMap* src_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    int vec_idx = blockIdx.x;
    kittens::tma::prefetch(shared_vector, src_tma_map, vec_idx);
}
```

### load_async

Asynchronously loads data from global memory into shared memory using TMA.

#### Tile Example:

```cuda
__global__ void kernel(const CUtensorMap* src_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    __shared__ kittens::tma::barrier bar;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::load_async(shared_tile, src_tma_map, bar, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const CUtensorMap* src_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    __shared__ kittens::tma::barrier bar;
    int vec_idx = blockIdx.x;
    kittens::tma::load_async(shared_vector, src_tma_map, bar, vec_idx);
}
```

### store_async

Asynchronously stores data from shared memory to global memory using TMA.

#### Tile Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::store_async(dst_tma_map, shared_tile, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    int vec_idx = blockIdx.x;
    kittens::tma::store_async(dst_tma_map, shared_vector, vec_idx);
}
```

### store_add_async

Asynchronously performs an element-wise addition and stores the result to global memory using TMA.

#### Tile Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::store_add_async(dst_tma_map, shared_tile, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    int vec_idx = blockIdx.x;
    kittens::tma::store_add_async(dst_tma_map, shared_vector, vec_idx);
}
```

### store_max_async

Asynchronously performs an element-wise maximum operation and stores the result to global memory using TMA.

#### Tile Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::store_max_async(dst_tma_map, shared_tile, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    int vec_idx = blockIdx.x;
    kittens::tma::store_max_async(dst_tma_map, shared_vector, vec_idx);
}
```

### store_min_async

Asynchronously performs an element-wise minimum operation and stores the result to global memory using TMA.

#### Tile Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::st_bf_2x1<> shared_tile;
    int tile_row_idx = blockIdx.y;
    int tile_col_idx = blockIdx.x;
    kittens::tma::store_min_async(dst_tma_map, shared_tile, tile_row_idx, tile_col_idx);
}
```

#### Vector Example:

```cuda
__global__ void kernel(CUtensorMap* dst_tma_map) {
    __shared__ kittens::sv_bf_2 shared_vector;
    int vec_idx = blockIdx.x;
    kittens::tma::store_min_async(dst_tma_map, shared_vector, vec_idx);
}
```

### init_barrier

Initializes a TMA barrier for synchronization.

```cuda
__global__ void kernel() {
    __shared__ kittens::tma::barrier bar;
    kittens::tma::init_barrier<kittens::st_bf_2x1<>, 4, 3>(bar); // barrier is set up to load 4*3=12 st_bf_2x1 tiles
}
```

This function initializes a barrier used for synchronizing TMA operations. The template parameter specifies the type of data being synchronized, which helps in automatically setting the expected number of bytes.

As a note, we have found it to be a troublesome to set the number of threads to wait on here, and just use this for memory. Synchronize threads separately.

### set_bytes

Sets the number of bytes expected at the TMA barrier.

```cuda
__global__ void kernel() {
    __shared__ kittens::tma::barrier bar;
    uint32_t bytes = sizeof(kittens::st_bf_2x1<>);
    kittens::tma::set_bytes(bar, bytes);
}
```

This function manually sets the number of bytes expected at the barrier. It's useful when the automatic byte calculation from `init_barrier` isn't sufficient.

### arrive_and_wait

Waits at a TMA barrier until the memory operation is complete and sufficient threads have arrived.

```cuda
__global__ void kernel() {
    __shared__ kittens::tma::barrier bar;
    int kPhaseBit = 0; // or 1, depending on the phase, which flips each time this is called.
    kittens::tma::arrive_and_wait(bar, kPhaseBit);
}
```

This function is used to synchronize threads at a barrier point. Threads will wait here until the specified memory operation is complete and the required number of threads have reached the barrier.

### store_commit_group

Commits previous asynchronous TMA stores to a group and performs them.

```cuda
__global__ void kernel() {
    // ... previous TMA store operations ...
    kittens::tma::store_commit_group();
}
```

This function is used to commit and execute a group of previously issued asynchronous TMA store operations. It ensures that all stores in the current group are executed.

### store_async_wait

Waits for previous committed TMA store groups to complete.

```cuda
template <int N = 0>
__global__ void kernel() {
    // ... previous TMA store operations and commits ...
    kittens::tma::store_async_wait<N>();
}
```

This function waits for the completion of previously committed TMA store groups. The template parameter `N` specifies the maximum number of remaining TMA store groups to wait for. By default, it waits for all groups to complete.

## Distributed Shared Memory (DSMEM) Operations

### distribute

Distributes data from a source shared tile or vector to a destination shared tile or vector across different thread blocks.

#### Tile Example:

```cuda
__global__ void kernel(int cluster_size) {
    __shared__ kittens::st_bf_2x1<> src_tile, dst_tile;
    __shared__ kittens::dsmem::barrier bar;
    int dst_idx = threadIdx.y;
    kittens::dsmem::distribute(dst_tile, src_tile, cluster_size, dst_idx, bar);
}
```

#### Vector Example:

```cuda
__global__ void kernel(int cluster_size) {
    __shared__ kittens::sv_bf_2 src_vector, dst_vector;
    __shared__ kittens::dsmem::barrier bar;
    int dst_idx = threadIdx.x;
    kittens::dsmem::distribute(dst_vector, src_vector, cluster_size, dst_idx, bar);
}
```

### init_barrier

Initializes a DSMEM barrier for synchronization across thread blocks.

```cuda
__global__ void kernel() {
    __shared__ kittens::dsmem::barrier bar;
    kittens::dsmem::init_barrier<kittens::st_bf_2x1<>>(bar);
}
```

### arrive_and_wait

Waits at a DSMEM barrier until the memory and sufficient threads have arrived.

```cuda
__global__ void kernel() {
    __shared__ kittens::dsmem::barrier bar;
    int kPhaseBit = 0; // or 1, depending on the phase
    kittens::dsmem::arrive_and_wait(bar, kPhaseBit);
}
```

### set_bytes

Sets the number of bytes expected at the DSMEM barrier.

```cuda
__global__ void kernel() {
    __shared__ kittens::dsmem::barrier bar;
    uint32_t bytes = sizeof(kittens::st_bf_2x1<>);
    kittens::dsmem::set_bytes(bar, bytes);
}
```

This documentation covers all the memory operations in ThunderKittens, including examples for both vector and tile operations where applicable. The operations are organized into sections for easy reference and understanding.