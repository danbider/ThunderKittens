#pragma once 

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"
#include "tma.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {

template<kittens::ducks::st::all ST, int num_tiles_, int num_tiles_dim_=0>
struct TileIterator {
    // for tma
    const CUtensorMap* tma_map_d; 
    int phase_bit;
    uint64_t* mem_barrier;
    bool first_load; 

    int tb_idx; // thread block idx

    // note how we differentiate between tiles and blocks (not sure if this is the best idea...)
    // tiles  == (e.g. (&q_smem)[number_tiles])
    // blocks == (e.g. kv_blocks = N / (NUM_WORKERS_KV*k_smem[0][0].rows)
    // why differentiate? because sometimes, number_tiles dimension != num_blocks dimension!

    static constexpr int number_tiles  = num_tiles_;     
    static constexpr int num_tiles_dim = num_tiles_dim_; // dimension which number_tiles strides (0 = rows, 1 = cols)
    int tile_idx;                                       // current idx in number_tiles
     
    int num_blocks;     // total number of times we need to load number_tiles
    int num_blocks_dim; // dimension of num_blocks (0 = rows, 1 = cols)
    int block_idx;      // current idx in num_blocks

    __device__ TileIterator(const CUtensorMap* tma_map, int num_blocks, int tb_idx, uint64_t* mem_barrier = nullptr, int num_blocks_dim=0)
    : phase_bit(0), num_blocks(num_blocks), num_blocks_dim(num_blocks_dim), block_idx(0), tb_idx(tb_idx), 
    first_load(true), mem_barrier(mem_barrier)
    {
        tma_map_d = tma_map;

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
        
            tma::init_barrier(*mem_barrier);
            tma::set_bytes(*mem_barrier, 
                size_bytes<ST> * number_tiles
            ); 

        }

        tile_idx = tb_idx * num_blocks; 
        tile_idx = (num_blocks_dim == num_tiles_dim) ? (tile_idx * number_tiles) : (tile_idx); 
    }

    __device__ void load_async(ST (&dst)[number_tiles], int custom_idx=-1) {
        static_assert(std::is_pointer_v<decltype(mem_barrier)>, "mem_barrier must be initialized in iterator");

        if (kittens::warpid() == 0) {
            if (first_load) {
                first_load = false;
            }
            else {
                tma::set_bytes(*mem_barrier, 
                    size_bytes<ST> * number_tiles
                );
            }
            
            for (int i = 0; i < number_tiles; i++) {
                custom_idx = (custom_idx == -1) ? (block_idx) : (custom_idx);
                int mod = (num_blocks_dim == num_tiles_dim) ? (custom_idx * number_tiles) : (custom_idx);
                if constexpr (num_tiles_dim == 0) { // number_tiles loads across rows
                    tma::load_async(dst[i], tma_map_d, *mem_barrier, tile_idx + mod + i); 
                }
                if constexpr (num_tiles_dim == 1) { // number_tiles loads across cols
                    tma::load_async(dst[i], tma_map_d, *mem_barrier, tile_idx + mod, i);
                }
            }
        }
    }

    __device__ void arrive_and_wait() {
        static_assert(std::is_pointer_v<decltype(mem_barrier)>, "mem_barrier must be initialized in iterator");
        
        tma::arrive_and_wait(*mem_barrier, phase_bit);
        phase_bit ^= 1;
    }

    __device__ void store_async(ST (&src)[number_tiles], int custom_idx=-1) {
        if (kittens::warpid() % 4) {
            custom_idx = (custom_idx == -1) ? (block_idx) : (custom_idx); 
            int mod = (num_blocks_dim == num_tiles_dim) ? (custom_idx * number_tiles) : (custom_idx); 
            if constexpr (num_tiles_dim == 0) {
                tma::store_async(const_cast<CUtensorMap*>(tma_map_d), src[kittens::warpgroupid()], tile_idx + mod + kittens::warpgroupid()); 
                tma::store_commit_group(); 
            }
            if constexpr (num_tiles_dim == 1) {
                tma::store_async(const_cast<CUtensorMap*>(tma_map_d), src[kittens::warpgroupid()], tile_idx + mod, kittens::warpgroupid()); 
                tma::store_commit_group(); 
            }
        }
    }

    __device__ TileIterator& operator++() {
        ++block_idx;
        return *this;
    }

    __device__ TileIterator operator++(int) {
        TileIterator tmp = *this;
        ++(*this);
        return tmp;
    }

    __device__ TileIterator& operator--() {
        --block_idx;
        return *this;
    }

    __device__ TileIterator operator--(int) {
        TileIterator tmp = *this;
        --(*this);
        return tmp;
    }

    __device__ bool hasNext() const {
        return block_idx < num_blocks;
    }
}; 

} // namespace kittens