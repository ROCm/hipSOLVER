/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver/hipsolver.h"

#ifdef __cplusplus
#include "complex.hpp"
#include "hipsolver_datatype2string.hpp"
#include <cmath>
#include <immintrin.h>
#include <random>
#include <type_traits>
#include <vector>
#endif

#include <stdio.h>
#include <stdlib.h>

/*!\file
 * \brief provide data initialization, timing, hipsolver type <-> lapack char conversion utilities.
 */

#ifdef GOOGLE_TEST

#include <gtest/gtest.h>

#define CHECK_HIP_ERROR(error) ASSERT_EQ(error, hipSuccess)

inline void hipsolver_expect_status(hipsolverStatus_t status, hipsolverStatus_t expected)
{
    if(status != HIPSOLVER_STATUS_NOT_SUPPORTED)
        ASSERT_EQ(status, expected);
}

#define EXPECT_ROCBLAS_STATUS(status, expected) hipsolver_expect_status(status, expected)
#define CHECK_ROCBLAS_ERROR(status) hipsolver_expect_status(status, HIPSOLVER_STATUS_SUCCESS)

#else

#define CHECK_HIP_ERROR(error)                    \
    do                                            \
    {                                             \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

inline void hipsolver_expect_status(hipsolverStatus_t status, hipsolverStatus_t expected)
{
    if(status != expected && status != HIPSOLVER_STATUS_NOT_SUPPORTED)
    {
        fprintf(stderr,
                "hipSOLVER status error: Expected: %s, Actual: %s\n",
                hipsolver2string_status(expected),
                hipsolver2string_status(status));
        if(expected == HIPSOLVER_STATUS_SUCCESS)
            exit(EXIT_FAILURE);
    }
}

#define EXPECT_ROCBLAS_STATUS(status, expected) hipsolver_expect_status(status, expected)
#define CHECK_ROCBLAS_ERROR(status) hipsolver_expect_status(status, HIPSOLVER_STATUS_SUCCESS)

#endif

#ifdef __cplusplus

/* ============================================================================================
 */
/*! \brief  local handle which is automatically created and destroyed  */
class hipsolver_local_handle
{
    hipsolverHandle_t m_handle;

public:
    hipsolver_local_handle()
    {
        hipsolverCreate(&m_handle);
    }
    ~hipsolver_local_handle()
    {
        hipsolverDestroy(m_handle);
    }

    hipsolver_local_handle(const hipsolver_local_handle&) = delete;
    hipsolver_local_handle(hipsolver_local_handle&&)      = delete;
    hipsolver_local_handle& operator=(const hipsolver_local_handle&) = delete;
    hipsolver_local_handle& operator=(hipsolver_local_handle&&) = delete;

    // Allow hipsolver_local_handle to be used anywhere hipsolverHandle_t is expected
    operator hipsolverHandle_t&()
    {
        return m_handle;
    }
    operator const hipsolverHandle_t&() const
    {
        return m_handle;
    }
};

/* ============================================================================================
 */

// Return true if value is NaN
template <typename T>
inline bool hipsolver_isnan(T)
{
    return false;
}
inline bool hipsolver_isnan(double arg)
{
    return std::isnan(arg);
}
inline bool hipsolver_isnan(float arg)
{
    return std::isnan(arg);
}
inline bool hipsolver_isnan(hipsolverComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}
inline bool hipsolver_isnan(hipsolverDoubleComplex arg)
{
    return std::isnan(arg.real()) || std::isnan(arg.imag());
}

/* =============================================================================================== */
/* Complex / real helpers.                                                                         */
template <typename T>
static constexpr bool is_complex = false;

/* Workaround for clang bug:
   https://bugs.llvm.org/show_bug.cgi?id=35863
*/
#if __clang__
#define HIPSOLVER_CLANG_STATIC static
#else
#define HIPSOLVER_CLANG_STATIC
#endif

template <>
HIPSOLVER_CLANG_STATIC constexpr bool is_complex<hipsolverComplex> = true;

template <>
HIPSOLVER_CLANG_STATIC constexpr bool is_complex<hipsolverDoubleComplex> = true;

// Get base types from complex types.
template <typename T, typename = void>
struct real_t_impl
{
    using type = T;
};

template <typename T>
struct real_t_impl<T, std::enable_if_t<is_complex<T>>>
{
    using type = decltype(T{}.real());
};

template <typename T>
using real_t = typename real_t_impl<T>::type;

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */

using hipsolver_rng_t = std::mt19937;
extern hipsolver_rng_t hipsolver_rng, hipsolver_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void hipsolver_seedrand()
{
    hipsolver_rng = hipsolver_seed;
}

class hipsolver_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(hipsolver_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(hipsolver_rng);
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN Complex
    explicit operator hipsolverComplex()
    {
        return {float(*this), float(*this)};
    }

    // Random NaN Double Complex
    explicit operator hipsolverDoubleComplex()
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return T(rand() % 10 + 1);
};

// for hipsolverComplex, generate 2 floats
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipsolverComplex random_generator<hipsolverComplex>()
{
    return hipsolverComplex(rand() % 10 + 1, rand() % 10 + 1);
    return {float(rand() % 10 + 1), float(rand() % 10 + 1)};
}

// for hipsolverDoubleComplex, generate 2 doubles
/*! \brief  generate two random numbers in range [1,2,3,4,5,6,7,8,9,10] */
template <>
inline hipsolverDoubleComplex random_generator<hipsolverDoubleComplex>()
{
    return hipsolverDoubleComplex(rand() % 10 + 1, rand() % 10 + 1);
    return {double(rand() % 10 + 1), double(rand() % 10 + 1)};
}

/*! \brief  generate a random number in range [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10] */
template <typename T>
inline T random_generator_negative()
{
    // return rand()/( (T)RAND_MAX + 1);
    return -T(rand() % 10 + 1);
};

// for complex, generate two values, convert both to negative
/*! \brief  generate a random real value in range [-1, -10] and random
*           imaginary value in range [-1, -10]
*/
template <>
inline hipsolverComplex random_generator_negative<hipsolverComplex>()
{
    return {float(-(rand() % 10 + 1)), float(-(rand() % 10 + 1))};
}

template <>
inline hipsolverDoubleComplex random_generator_negative<hipsolverDoubleComplex>()
{
    return {double(-(rand() % 10 + 1)), double(-(rand() % 10 + 1))};
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief Packs strided_batched matricies into groups of 4 in N */
template <typename T>
void hipsolver_packInt8(
    std::vector<T>& A, size_t M, size_t N, size_t lda, size_t batch_count = 1, size_t stride_a = 0)
{
    std::vector<T> temp(A);
    for(size_t b = 0; b < batch_count; b++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * b)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * b)];
}

/* ============================================================================================ */

/* ============================================================================================ */
/*! \brief  turn float -> 's', double -> 'd', hipsolverComplex -> 'c', hipsolverDoubleComplex
 * -> 'z' */
template <typename T>
char type2char();

/* ============================================================================================ */
/*! \brief  turn float -> int, double -> int, hipsolverComplex.real() -> int,
 * hipsolverDoubleComplex.real() -> int */
template <typename T>
int type2int(T val);

/* ============================================================================================ */
/*! \brief  Debugging purpose, print out CPU and GPU result matrix, not valid in complex number  */
template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
void print_matrix(T* CPU_result, T* GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=%.8g, GPU result=%.8g\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda]),
                   double(GPU_result[j + i * lda]));
}

/*! \brief  Debugging purpose, print out CPU and GPU result matrix, valid for complex number  */
template <typename T, std::enable_if_t<+is_complex<T>, int> = 0>
void print_matrix(T* CPU_result, T* GPU_result, int m, int n, int lda)
{
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            printf("matrix  col %d, row %d, CPU result=(%.8g,%.8g), GPU result=(%.8g,%.8g)\n",
                   i,
                   j,
                   double(CPU_result[j + i * lda].real()),
                   double(CPU_result[j + i * lda].imag()),
                   double(GPU_result[j + i * lda].real()),
                   double(GPU_result[j + i * lda].imag()));
}

/* =============================================================================================== */

/* ============================================================================================ */

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipsolver sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us();

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

/*! \brief  CPU Timer(in microsecond): no GPU synchronization and return wall time */
double get_time_us_no_sync();

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */
