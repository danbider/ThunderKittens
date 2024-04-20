#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace detail {

// templated wrapper for PTX
template<int width, int trans_b>
struct wgmma_base {
    __device__ static inline void rt_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const rt_bf<1, 1, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc
    );
    __device__ static inline void st_st(
        rt_fl<1, width, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc
    );
};

#include "4x1.impl"
#include "4x2.impl"
#include "4x3.impl"
#include "4x4.impl"

// can add bigger ones later
// #include "4x5.impl"
// #include "4x6.impl"
// #include "4x7.impl"
#include "4x8.impl"

}
}