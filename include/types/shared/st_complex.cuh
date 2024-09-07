/**
 * @file
 * @brief Abstraction for a complex register tile composed of real and imaginary tiles
 */
 
#pragma once

#include "st.cuh"

namespace kittens {

namespace ducks {
namespace st {
/**
 * @brief A dummy type used to identify complex register tiles.
 * 
 * For a type to quack like an st_cmplx, it should define its identifier as ducks::st::cmplx_identifier.
 * If a type quacks like ducks::st::cmplx_identifier, it will be treated as an st_cmplx by compiler checks.
 */
struct cmplx_identifier {};
} // namespace st
} // namespace ducks

/**
 * @brief Complex tile structure
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _height The height of the tile in terms of the number of subtiles.
 * @tparam _width The width of the tile in terms of the number of subtiles.
 * @tparam _layout The layout of the internal register tiles
 *
 * This structure is designed to abstract complex number operations internally to the real and imaginary
 * shared tiles, respectively
 * 
 *
 */
template<typename _T, int _height, int _width>
struct st_cmplx {
    using identifier = ducks::st::cmplx_identifier;
    using dtype = st<_T, _height, _width>; /// Data type of each internal tile.

    // Real/imag tiles have same internal layout and size
    dtype real;
    dtype imag;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace st {

/**
* @brief Concept for shared tiles that are complex.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a shared tile.
* - T has a complex tile identifier.
*/
template <typename T> concept complex = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, cmplx_identifier> && all<typename T::dtype>;

}
}

template<int _height, int _width> using st_cmplx_bf = st_cmplx<bf16, _height, _width>;



}