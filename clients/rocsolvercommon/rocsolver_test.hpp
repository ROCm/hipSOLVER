/* ************************************************************************
 * Copyright 2018-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <cstdarg>
#include <iomanip>
#include <limits>
#include <sstream>

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

template <typename T>
constexpr double get_safemin()
{
    using S  = decltype(std::real(T{}));
    auto eps = get_epsilon<S>();
    auto s1  = std::numeric_limits<S>::min();
    auto s2  = 1 / std::numeric_limits<S>::max();
    if(s2 > s1)
        return s2 * (1 + eps);
    return s1;
}

#ifdef GOOGLE_TEST
#define ROCSOLVER_TEST_CHECK(T, max_error, tol) ASSERT_LE((max_error), (tol)*get_epsilon<T>())
#else
#define ROCSOLVER_TEST_CHECK(T, max_error, tol)
#endif

typedef enum rocsolver_inform_type_
{
    inform_quick_return,
    inform_invalid_size,
    inform_invalid_args,
    inform_mem_query,
} rocsolver_inform_type;

inline void rocsolver_bench_inform(rocsolver_inform_type it, size_t arg = 0)
{
    switch(it)
    {
    case inform_quick_return:
        printf("Quick return...\n");
        break;
    case inform_invalid_size:
        printf("Invalid size arguments...\n");
        break;
    case inform_invalid_args:
        printf("Invalid value in arguments...\n");
        break;
    case inform_mem_query:
        printf("%li bytes of device memory are required...\n", arg);
        break;
    }
    printf("No performance data to collect.\n");
    printf("No computations to verify.\n");
    std::fflush(stdout);
}

inline void rocsolver_bench_output()
{
    // empty version
    std::cerr << std::endl;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    std::stringstream ss;
    ss << std::left << std::setw(15) << arg;

    std::cerr << ss.str();
    if(sizeof...(Ts) > 0)
        std::cerr << ' ';
    rocsolver_bench_output(args...);
}

// template <typename T, std::enable_if_t<!is_complex<T>, int> = 0>
// inline T sconj(T scalar)
// {
//     return scalar;
// }

// template <typename T, std::enable_if_t<is_complex<T>, int> = 0>
// inline T sconj(T scalar)
// {
//     return std::conj(scalar);
// }

// // A struct implicity convertable to and from char, used so we can customize
// // Google Test printing for LAPACK char arguments without affecting the default
// // char output.
// struct rocsolver_op_char
// {
//     rocsolver_op_char(char c)
//         : data(c)
//     {
//     }

//     operator char() const
//     {
//         return data;
//     }

//     char data;
// };

// // gtest printers

// inline std::ostream& operator<<(std::ostream& os, rocblas_status x)
// {
//     return os << rocblas_status_to_string(x);
// }

// inline std::ostream& operator<<(std::ostream& os, rocsolver_op_char x)
// {
//     return os << x.data;
// }
