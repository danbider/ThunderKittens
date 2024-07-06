# ThunderKittens WGMMA Operations Documentation

This document provides detailed information about the Warp Group Matrix Multiply-Accumulate (WGMMA) operations in ThunderKittens, including examples and layout requirements.

An important note: *kittens::warpgroup and kittens::group<4> are aliases and are used interchangeably.*

## Table of Contents
1. [Matrix Multiply-Accumulate Operations](#matrix-multiply-accumulate-operations)
2. [Matrix Multiply Operations](#matrix-multiply-operations)
3. [Synchronization and Control Operations](#synchronization-and-control-operations)

## Matrix Multiply-Accumulate Operations

### mma_AB

Performs matrix multiply-accumulate operation (C = A * B + C).

#### Allowed Layouts:
- A: Register tile (row-major) or shared memory tile (wgmma_swizzle or wgmma_interleave)
- B: Shared memory tile (wgmma_swizzle or wgmma_interleave)
- C: Register tile (row-major)

#### Example:

```cuda
kittens::rt_bf_1x4<kittens::ducks::rt_layout::row> a;
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_swizzle> b;
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mma_AB(c, a, b);
```

### mma_ABt

Performs matrix multiply-accumulate operation with transposed B (C = A * B^T + C).

#### Allowed Layouts:
- A: Register tile (row-major) or shared memory tile (wgmma_swizzle or wgmma_interleave)
- B: Shared memory tile (wgmma_swizzle or wgmma_interleave)
- C: Register tile (row-major)

#### Example:

```cuda
kittens::rt_bf_1x4<kittens::ducks::rt_layout::row> a;
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_swizzle> b;
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mma_ABt(c, a, b);
```

### mma_AtB

Performs matrix multiply-accumulate operation with transposed A (C = A^T * B + C).

#### Allowed Layouts:
- A: Shared memory tile (wgmma_interleave)
- B: Shared memory tile (wgmma_interleave)
- C: Register tile (row-major)

#### Example:

```cuda
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_interleave> a, b;
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mma_AtB(c, a, b);
```

### mma_AtBt

Performs matrix multiply-accumulate operation with both A and B transposed (C = A^T * B^T + C).

#### Allowed Layouts:
- A: Shared memory tile (wgmma_interleave)
- B: Shared memory tile (wgmma_swizzle or wgmma_interleave)
- C: Register tile (row-major)

#### Example:

```cuda
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_interleave> a;
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_swizzle> b;
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mma_AtBt(c, a, b);
```

## Matrix Multiply Operations

These operations are similar to their MMA counterparts but they reset the result (C = A * B) instead of accumulating it.

### mm_AB, mm_ABt, mm_AtB, mm_AtBt

These functions have the same layout requirements as their MMA counterparts; the only difference is how they overwrite the C registers.

#### Example:

```cuda
kittens::rt_bf_1x4<kittens::ducks::rt_layout::row> a;
kittens::st_bf_4x4<kittens::ducks::st_layout::wgmma_swizzle> b;
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mm_AB(c, a, b);
```

## Synchronization and Control Operations

### mma_fence

Synchronizes the warp group and ensures that all writes to shared memory are visible.

#### Example:

```cuda
kittens::rt_fl_1x4<kittens::ducks::rt_layout::row> c;
kittens::group<4>::mma_fence(c);
```

### mma_commit_group

Commits the current set of warp group matrix multiply accumulate calls.

#### Example:

```cuda
kittens::group<4>::mma_commit_group();
```

### mma_async_wait

Waits for the warp group to reach a synchronization point. A template parameter can be set, to allow a max number of groups to remain in the pipeline.

#### Example:

```cuda
kittens::group<4>::mma_async_wait<2>(); // waits until no more than 2 committed mma_async groups are active.
kittens::group<4>::mma_async_wait(); // waits until all active mma_async groups have finished.
```

This documentation covers the WGMMA operations in ThunderKittens, including examples and layout requirements for each operation. The operations are organized into sections for easy reference and understanding.