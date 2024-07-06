# ThunderKittens Group Memory Operations Documentation

This document provides detailed information about the group memory operations in ThunderKittens, including examples for both vector and tile operations where applicable.

## Table of Contents
1. [Global to Register](#global-to-register)
2. [Global to Shared](#global-to-shared)
3. [Shared to Register](#shared-to-register)

## Global to Register

### load

Collaboratively loads data from global memory into register tiles or vectors split across a group.

#### Tile Example:

```cuda
// This implicitly loads an 8x1 tile, split by rows across the 4 warps of the group.
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    kittens::rt_fl_2x1<> my_tile;
    group_4::load(my_tile, global_data, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    kittens::row_vec<kittens::rt_fl_2x4<>> my_vector;
    group_4::load(my_vector, global_data);
}
```

### store

Collaboratively stores data from register tiles or vectors split across a group to global memory.

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    kittens::rt_fl_2x1<> my_tile;
    // ... operations on my_tile ...
    group_4::store(global_data, my_tile, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    kittens::row_vec<kittens::rt_fl_2x4<>> my_vector;
    // ... operations on my_vector ...
    group_4::store(global_data, my_vector);
}
```

## Global to Shared

### load

Collaboratively loads data from global memory into a shared memory tile or vector. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    group_4::load(shared_tile, global_data, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    group_4::load(shared_vector, global_data);
}
```

### store

Collaboratively stores data from a shared memory tile or vector to global memory. Can convert types on the fly.

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    // ... operations on shared_tile ...
    group_4::store(global_data, shared_tile, row_stride);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    // ... operations on shared_vector ...
    group_4::store(global_data, shared_vector);
}
```

### load_async

Asynchronously loads data from global memory into a shared memory tile or vector using CUDA's asynchronous copy mechanism.

#### Tile Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    group_4::load_async(shared_tile, global_data, row_stride, barrier);
}
```

#### Vector Example:

```cuda
__global__ void kernel(const kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    group_4::load_async(shared_vector, global_data, barrier);
}
```

### store_async

Asynchronously stores data from a shared memory tile or vector to global memory using CUDA's asynchronous copy mechanism.

#### Tile Example:

```cuda
__global__ void kernel(kittens::bf16* global_data, int row_stride) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    // ... operations on shared_tile ...
    group_4::store_async(global_data, shared_tile, row_stride, barrier);
}
```

#### Vector Example:

```cuda
__global__ void kernel(kittens::bf16* global_data) {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    // ... operations on shared_vector ...
    group_4::store_async(global_data, shared_vector, barrier);
}
```

## Shared to Register

### load

Collaboratively loads data from a shared memory tile or vector into register tiles or vectors split across a group.

#### Tile Example:

```cuda
__global__ void kernel() {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    kittens::rt_fl_2x1<> my_tile;
    group_4::load(my_tile, shared_tile);
}
```

#### Vector Example:

```cuda
__global__ void kernel() {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    kittens::col_vec<kittens::rt_fl_2x4<>> my_vector;
    group_4::load(my_vector, shared_vector);
}
```

### store

Collaboratively stores data from register tiles or vectors split across a group into a shared memory tile or vector.

#### Tile Example:

```cuda
__global__ void kernel() {
    using group_4 = kittens::group<4>;
    __shared__ kittens::st_bf_8x1<> shared_tile;
    kittens::rt_fl_2x1<> my_tile;
    // ... operations on my_tile ...
    group_4::store(shared_tile, my_tile);
}
```

#### Vector Example:

```cuda
__global__ void kernel() {
    using group_4 = kittens::group<4>;
    __shared__ kittens::sv_bf_8 shared_vector;
    kittens::col_vec<kittens::rt_fl_2x4<>> my_vector;
    // ... operations on my_vector ...
    group_4::store(shared_vector, my_vector);
}
```

This documentation covers the group memory operations in ThunderKittens, including examples for both vector and tile operations where applicable. The operations are organized into sections for easy reference and understanding.