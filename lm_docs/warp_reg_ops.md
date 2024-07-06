# ThunderKittens Warp Register Operations Documentation

This document provides detailed information about the warp register operations in ThunderKittens, including examples for both vector and tile operations where applicable.

## Table of Contents
1. [Layout and Conversion Operations](#layout-and-conversion-operations)
2. [Reduction Operations](#reduction-operations)
3. [Matrix Multiply-Accumulate Operations](#matrix-multiply-accumulate-operations)
4. [Mapping Operations](#mapping-operations)
5. [Mathematical Operations](#mathematical-operations)

## Layout and Conversion Operations

### swap_layout

Swaps the layout of a register tile between row-major and column-major.

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::row> src_tile;
kittens::rt_bf_2x2<ducks::rt_layout::col> dst_tile;
kittens::swap_layout(dst_tile, src_tile);
```

### swap_layout_inplace

Swaps the layout of a register tile in-place.

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::row> tile;
kittens::swap_layout_inplace(tile);
```

### transpose_sep

Transposes a register tile. Note that the destination and source MUST be distinct objects, otherwise this will silently fail.

#### Example:

```cuda
kittens::rt_bf_1x2<> src_tile;
kittens::rt_bf_2x1<> dst_tile;
kittens::transpose_sep(dst_tile, src_tile);
```

### transpose_inplace

Transposes a square register tile in-place.

#### Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::transpose_inplace(tile);
```

### copy

Copies data from one register tile or vector to another, potentially converting the underlying type.

#### Tile Example:

```cuda
kittens::rt_bf_2x2<> src_tile;
kittens::rt_fl_2x2<> dst_tile;
kittens::copy(dst_tile, src_tile);
```

#### Vector Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::row_vec<decltype(tile)> src_vec;
kittens::col_vec<decltype(tile)> dst_vec;
kittens::copy(dst_vec, src_vec);
```

### make_causal

Makes a square register tile causal by zeroing elements above the main diagonal.

#### Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::make_causal(tile, tile);
```

### subtile_inplace

Returns a reference to a subtile of the given tile.

#### Example:

```cuda
kittens::rt_bf_4x2<> tile;
auto subtile = kittens::subtile_inplace<2>(tile, 0);
```

## Reduction Operations

### row_reduce

Performs a row-wise reduction on a matrix.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile;
kittens::col_vec<decltype(src_tile)> row_accum;
kittens::row_reduce<base_ops::sum, decltype(row_accum), decltype(src_tile), true>(row_accum, src_tile, row_accum);
```

### col_reduce

Performs a column-wise reduction on a matrix.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile;
kittens::row_vec<decltype(src_tile)> col_accum;
kittens::col_reduce<base_ops::sum, decltype(col_accum), decltype(src_tile), true>(col_accum, src_tile, col_accum);
```

### row_max, row_min, row_sum, row_prod

Performs specific row-wise reductions (maximum, minimum, sum, product) on a matrix.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile;
kittens::col_vec<decltype(src_tile)> row_accum;
kittens::row_max(row_accum, src_tile, row_accum); // starts from current row_accum values
```

### col_max, col_min, col_sum, col_prod

Performs specific column-wise reductions (maximum, minimum, sum, product) on a matrix.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile;
kittens::row_vec<decltype(src_tile)> col_accum;
kittens::col_sum(col_accum, src_tile); // ignores current col_accum values
```

### reduce (for vectors)

Performs a reduction operation on elements of a register vector within a warp.

#### Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::row_vec<decltype(tile)> src_vec;
float result;
kittens::reduce<base_ops::sum>(result, src_vec, 0.0f);
```

## Matrix Multiply-Accumulate Operations

### mma_AB

Performs matrix multiply-accumulate operation (C = A * B + C).

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::row> a;
kittens::rt_bf_2x2<ducks::rt_layout::col> b;
kittens::rt_fl_2x2<ducks::rt_layout::row> c, d;
kittens::mma_AB(d, a, b, c);
```

### mma_ABt

Performs matrix multiply-accumulate operation with transposed B (C = A * B^T + C).

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::row> a, b;
kittens::rt_fl_2x2<ducks::rt_layout::row> c, d;
kittens::mma_ABt(d, a, b, c);
```

### mma_AtB

Performs matrix multiply-accumulate operation with transposed A (C = A^T * B + C).

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::col> a;
kittens::rt_bf_2x2<ducks::rt_layout::col> b;
kittens::rt_fl_2x2<ducks::rt_layout::row> c, d;
kittens::mma_AtB(d, a, b, c);
```

### mma_AtBt

Performs matrix multiply-accumulate operation with both A and B transposed (C = A^T * B^T + C).

#### Example:

```cuda
kittens::rt_bf_2x2<ducks::rt_layout::col> a;
kittens::rt_bf_2x2<ducks::rt_layout::row> b;
kittens::rt_fl_2x2<ducks::rt_layout::row> c, d;
kittens::mma_AtBt(d, a, b, c);
```

## Mapping Operations

### unary_map

Applies a unary operation to each element of a tile.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::unary_map<base_ops::exp>(dst_tile, src_tile);
```

### bin_map

Applies a binary operation to each element of a tile with another tile or a scalar.

#### Example:

```cuda
kittens::rt_bf_2x2<> lhs_tile, rhs_tile, dst_tile;
kittens::bin_map<base_ops::sum>(dst_tile, lhs_tile, rhs_tile);
```

### row_map

Applies an operation across the rows of a tile.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::col_vec<decltype(src_tile)> row_values;
kittens::row_map<base_ops::sum>(dst_tile, src_tile, row_values);
```

### col_map

Applies an operation across the columns of a tile.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::row_vec<decltype(src_tile)> col_values;
kittens::col_map<base_ops::sum>(dst_tile, src_tile, col_values);
```

## Mathematical Operations

### zero, one, pos_infty, neg_infty

Sets all elements of a tile or vector to zero, one, positive infinity, or negative infinity.

#### Tile Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::zero(tile);
```

#### Vector Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::row_vec<decltype(tile)> vec;
kittens::zero(vec);
```

### exp, log, abs, relu

Applies exponential, logarithm, absolute value, or ReLU function to each element of a tile or vector.

#### Tile Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::exp(dst_tile, src_tile);
```

#### Vector Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::row_vec<decltype(tile)> src_vec, dst_vec;
kittens::exp(dst_vec, src_vec);
```

### max, min, add, sub, mul, div

Performs element-wise maximum, minimum, addition, subtraction, multiplication, or division on tiles or vectors.

#### Tile Example:

```cuda
kittens::rt_bf_2x2<> lhs_tile, rhs_tile, dst_tile;
kittens::add(dst_tile, lhs_tile, rhs_tile);
```

#### Vector Example:

```cuda
kittens::rt_bf_2x2<> tile;
kittens::row_vec<decltype(tile)> lhs_vec, rhs_vec, dst_vec;
kittens::add(dst_vec, lhs_vec, rhs_vec);
```

### add_row, sub_row, mul_row, div_row

Performs row-wise addition, subtraction, multiplication, or division on a tile.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::col_vec<decltype(src_tile)> row_values;
kittens::add_row(dst_tile, src_tile, row_values);
```

### add_col, sub_col, mul_col, div_col

Performs column-wise addition, subtraction, multiplication, or division on a tile.

#### Example:

```cuda
kittens::rt_bf_2x2<> src_tile, dst_tile;
kittens::row_vec<decltype(src_tile)> col_values;
kittens::add_col(dst_tile, src_tile, col_values);
```

### broadcast_row, broadcast_col

Broadcasts a vector into a tile's rows or columns.

#### Example:

```cuda
kittens::rt_bf_2x2<> dst_tile;
kittens::col_vec<decltype(dst_tile)> row_values;
kittens::broadcast_row(dst_tile, row_values);
```

This documentation provides a comprehensive overview of the ThunderKittens Warp Register Operations, including examples for both vector and tile operations where applicable. The operations that exist for both tiles and vectors have been combined with examples for each type.