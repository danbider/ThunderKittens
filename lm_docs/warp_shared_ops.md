# ThunderKittens Warp Shared Memory Operations Documentation

This document provides detailed information about the warp shared memory operations in ThunderKittens, including examples for both vector and tile operations where applicable.

## Table of Contents
1. [Conversion Operations](#conversion-operations)
2. [Mapping Operations](#mapping-operations)
3. [Reduction Operations](#reduction-operations)

## Conversion Operations

### copy

Copies data from one shared memory structure to another, potentially with different data types and layouts. Applies to both tiles and vectors.

#### Tile Example:

```cuda
__shared__ kittens::st_bf_2x2<ducks::st_layout::row> src_tile;
__shared__ kittens::st_fl_2x2<ducks::st_layout::col> dst_tile;
kittens::copy(dst_tile, src_tile);
```

#### Vector Example:

```cuda
__shared__ kittens::sv_bf_2 src_vec;
__shared__ kittens::sv_fl_2 dst_vec;
kittens::copy(dst_vec, src_vec);
```

### subtile_inplace

Returns a reference to a subtile of the given shared tile.

#### Example:

```cuda
__shared__ kittens::st_bf_4x4<> tile;
auto subtile = kittens::subtile_inplace<2, 2>(tile, 0, 0);
```

### subvec_inplace

Returns a reference to a subvec of a given shared vector.

#### Example:

```cuda
__shared__ kittens::sv_bf_4 vec;
auto subvec = kittens::subvec_inplace<2>(vec, 0);
```

## Mapping Operations

### unary_map (for tiles) / unary_op (for vectors)

Applies a unary operation to each element of a tile or vector.

#### Tile Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile, dst_tile;
kittens::unary_map<base_ops::exp>(dst_tile, src_tile);
```

#### Vector Example:

```cuda
__shared__ kittens::sv_bf_2 src_vec, dst_vec;
kittens::unary_op<base_ops::exp>(dst_vec, src_vec);
```

### bin_map (for tiles) / bin_op (for vectors)

Applies a binary operation to each element of a tile or vector with another tile/vector or a scalar.

#### Tile Example:

```cuda
__shared__ kittens::st_bf_2x2<> lhs_tile, rhs_tile, dst_tile;
kittens::bin_map<base_ops::sum>(dst_tile, lhs_tile, rhs_tile);
```

#### Vector Example:

```cuda
__shared__ kittens::sv_bf_2 lhs_vec, rhs_vec, dst_vec;
kittens::bin_op<base_ops::sum>(dst_vec, lhs_vec, rhs_vec);
```

### row_map

Applies an operation across the rows of a tile.

#### Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile, dst_tile;
__shared__ kittens::sv_bf_2 row_values;
kittens::row_map<base_ops::sum>(dst_tile, src_tile, row_values);
```

### col_map

Applies an operation across the columns of a tile.

#### Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile, dst_tile;
__shared__ kittens::sv_bf_2 col_values;
kittens::col_map<base_ops::sum>(dst_tile, src_tile, col_values);
```

### Mathematical Operations

The following operations are available for both tiles and vectors:

Zero source arguments:
- `zero`: Sets all elements to zero.
- `one`: Sets all elements to one.
- `pos_infty`: Sets all elements to positive infinity.
- `neg_infty`: Sets all elements to negative infinity.
One source arg:
- `exp`: Applies exponential function.
- `log`: Applies natural logarithm.
- `abs`: Applies absolute value.
- `relu`: Applies rectified linear unit function.
Either two objects or broadcast with a scalar:
- `max`: Element-wise maximum.
- `min`: Element-wise minimum.
- `add`: Element-wise addition.
- `sub`: Element-wise subtraction.
- `mul`: Element-wise multiplication.
- `div`: Element-wise division.

#### Tile Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile, dst_tile;
kittens::exp(dst_tile, src_tile);
```

#### Vector Example:

```cuda
__shared__ kittens::sv_bf_2 src_vec, dst_vec;
kittens::add(dst_vec, src_vec, 1.0f); // broadcast with a scalar
```

### Tile-specific Operations

The following operations are specific to tiles:

- `add_row`: Adds row values to each row of a tile.
- `sub_row`: Subtracts row values from each row of a tile.
- `mul_row`: Multiplies each row of a tile by row values.
- `div_row`: Divides each row of a tile by row values.
- `broadcast_row`: Broadcasts a vector into a tile's rows.
- `add_col`: Adds column values to each column of a tile.
- `sub_col`: Subtracts column values from each column of a tile.
- `mul_col`: Multiplies each column of a tile by column values.
- `div_col`: Divides each column of a tile by column values.
- `broadcast_col`: Broadcasts a vector into a tile's columns.

#### Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile, dst_tile;
__shared__ kittens::sv_bf_2 row_values;
kittens::add_row(dst_tile, src_tile, row_values);
```

## Reduction Operations

### reduce (for vectors)

Performs a reduction operation on elements of a shared memory vector within a warp.

#### Example:

```cuda
__shared__ kittens::sv_bf_2 src_vec;
bf16 result;
kittens::reduce<base_ops::sum>(result, src_vec, 0.0f);
```

### row_reduce (for tiles)

Performs row-wise reduction on a matrix using a specified operation.

#### Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile;
__shared__ kittens::sv_bf_2 row_accum;
kittens::row_reduce<base_ops::sum>(row_accum, src_tile, row_accum);
```

### col_reduce (for tiles)

Performs column-wise reduction on a matrix using a specified operation.

#### Example:

```cuda
__shared__ kittens::st_bf_2x2<> src_tile;
__shared__ kittens::sv_bf_2 col_accum;
kittens::col_reduce<base_ops::sum>(col_accum, src_tile, col_accum);
```

### Common Reduction Operations

The following reduction operations are available for both tiles (row-wise and column-wise) and vectors:

- `max`: Finds the maximum value.
- `min`: Finds the minimum value.
- `sum`: Computes the sum of values.
- `prod`: Computes the product of values.

#### Tile Example (row-wise):

```cuda
__shared__ kittens::st_bf_2x2<> src_tile;
__shared__ kittens::sv_bf_2 row_accum;
kittens::row_max(row_accum, src_tile);
```

#### Vector Example:

```cuda
__shared__ kittens::sv_bf_2 src_vec;
bf16 result;
kittens::sum(result, src_vec);
```

This documentation provides a comprehensive overview of the ThunderKittens Warp Shared Memory Operations, including examples for both vector and tile operations where applicable.