# ThunderKittens Key Type Documentation

## 1. Register Tile (rt)

### Overview
The Register Tile (rt) is a fundamental structure in ThunderKittens for managing matrix data in GPU registers. It's designed to work efficiently with CUDA's tensor cores and is distributed across a warp of 32 threads.

### Key Features
- Templated structure allowing for different data types (16 or 32 bit), dimensions, and layouts
- Supports both row-major and column-major layouts
- Built on smaller 16x16 subtiles (rt_base)
- Provides nested types for row and column vectors

### Usage
The rt structure is primarily used for high-performance matrix operations that leverage tensor cores. It's particularly useful for operations like matrix multiplication, where data locality and efficient register usage are crucial.

### Benefits
- Optimized for tensor core operations
- Flexible dimensions allow for various matrix sizes
- Efficient register utilization across a warp
- Abstracts complex register layout details from the user

### Disadvantages
- Complexity in understanding the underlying data distribution
- May lead to high register pressure for larger tiles

### Important Details
- Limited to sizes that are multiples of 16x16 subtiles
- Data is distributed across 32 threads in a warp
- The internal layout is optimized for tensor core operations
- Users should be aware of the trade-offs between tile size and register pressure

```
// Register Tile (rt) declaration
__device__ void register_tile_declaration_example() {
    // Declare a 2x2 register tile of float2 elements (row-major layout)
    kittens::rt_fl_2x2 my_rt;

    // Declare a 4x1 register tile of bf16_2 elements (column-major layout)
    kittens::rt_bf_4x1<kittens::ducks::rt_layout::col> my_col_rt;
}
```

## 2. Register Vector (rv)

### Overview
The Register Vector (rv) is a structure designed to work alongside register tiles, providing a way to manipulate vector data efficiently within the register file.

### Key Features
- Templated for different data types and dimensions
- Supports both inner and outer dimensions for flexible layouts
- Designed to match the layout of register tiles for efficient operations

### Usage
rv structures are typically used for vector operations in conjunction with register tiles. They're particularly useful for operations like matrix-vector multiplication or accumulating results along rows or columns of a matrix.

### Benefits
- Efficient vector representation that aligns with register tile layouts
- Allows for vectorized operations across a warp
- Flexible dimensioning to match various matrix sizes

### Disadvantages
- Has redundant data storage to maintain alignment with tile layouts
- Can be less intuitive to work with compared to standard vector representations -- optimized for tiles; slow for 1D ops.

### Important Details
- The layout is designed to match rt structures for efficient combined operations
- The outer_dim and inner_dim parameters allow for flexible data arrangements within the vector
```
// Register Vector (rv) declaration
__device__ void register_vector_declaration_example() {
    // Declare a register vector with 2 outer dimensions and 1 inner dimension of float2
    kittens::rv<float2, 2, 1> my_rv;

    // Declare a register vector with 4 outer dimensions and 2 inner dimensions of bf16_2
    kittens::rv<kittens::bf16_2, 4, 2> my_bf_rv;

    // !! Preferred way !!: Declare a row vector from a register tile
    kittens::row_vec<kittens::rt_fl_4x2<>> my_row_vec;

    // !! Preferred way !!: Declare a column vector from a register tile
    kittens::col_vec<kittens::rt_bf_2x4<>> my_col_vec;
}

## 3. Shared Tile (st)

### Overview
The Shared Tile (st) is a structure for managing matrix data in CUDA shared memory. It supports various memory layouts and swizzling patterns to optimize memory access patterns.

### Key Features
- Templated for different data types, dimensions, and memory layouts
- Supports various swizzling modes for optimized memory access
- Provides subtile functionality for working with portions of the tile

### Usage
st structures are used when data needs to be shared across multiple threads or thread blocks. They're particularly useful for collaborative computations or as intermediate storage in multi-stage algorithms.

### Benefits
- Optimized memory layouts to reduce bank conflicts
- Flexible swizzling options for different access patterns
- Subtile support for working with tile subsets efficiently

### Disadvantages
- Complex indexing schemes can be difficult to understand and debug -- use the templated form [{row, col}] instead!
- Swizzled layouts may complicate direct memory access patterns

### Important Details
- Different layout options (naive, swizzle, wgmma_swizzle, wgmma_interleave) for various use cases
- Swizzling can significantly impact performance by reducing bank conflicts
- Use swizzled layouts where possible, and use wgmma_interleave only when a shared memory matrix needs to be transposed en route to a warpgroup matrix multiply
- The subtile functionality allows for efficient work distribution across thread blocks

```
// Shared Tile (st) declaration
__global__ void shared_tile_declaration_example(int num_workers) {
    extern __shared__ alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Declare a 1x1 shared tile of bf16 elements with swizzle layout
    using QK_BLOCK = kittens::st_bf_1x1<kittens::ducks::st_layout::swizzle>;
    QK_BLOCK (&q_s)[2][NUM_WORKERS] = al.allocate<QK_BLOCK, 2, NUM_WORKERS>();

    // Declare a 2x4 shared tile of bf16 elements with wgmma_swizzle layout
    using LARGER_BLOCK = kittens::st_bf_2x4<kittens::ducks::st_layout::wgmma_swizzle>;
    LARGER_BLOCK &larger_tile = al.allocate<LARGER_BLOCK>();
}
```

## 4. Shared Vector (sv)

### Overview
The Shared Vector (sv) is a simple vector structure for shared memory, designed to work alongside shared tiles for vector operations.

### Key Features
- Templated for different data types and lengths
- Simple, uniform layout in memory
- Provides subvector functionality

### Usage
sv structures are used for vector operations in shared memory, often in conjunction with shared tiles. They're useful for accumulating results or storing intermediate vector data that needs to be accessed by multiple threads.

### Benefits
- Simple, straightforward memory layout
- Easy to use and understand
- Efficient for collaborative vector operations across threads

### Important Details
- The layout is a simple array in memory, making it easy to work with
- Subvector functionality allows for efficient collaborative work across thread groups

```
// Shared Vector (sv) declaration
__global__ void shared_vector_declaration_example() {
    extern __shared__ alignment_dummy __shm[];
    kittens::shared_allocator al((int*)&__shm[0]);

    // Declare a shared vector of 2 tiles of bf16 elements
    kittens::sv_bf_2 &my_sv = al.allocate<kittens::sv_bf_2>();

    // !! Preferred way !!: Declare a shared vector from a shared tile
    kittens::row_vec<typeof(my_shared_tile)> my_row_vec;

    // !! Preferred way !!: Declare a column vector from a register tile
    kittens::col_vec<typeof(my_shared_tile)> my_row_vec;
}
```