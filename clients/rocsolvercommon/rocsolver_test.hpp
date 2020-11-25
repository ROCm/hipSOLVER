/* ************************************************************************
 * Copyright 2018-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef S_TEST_H
#define S_TEST_H

#include <boost/format.hpp>
#include <cstdarg>
#include <limits>

#define ROCSOLVER_BENCH_INFORM(case)                                    \
    do                                                                  \
    {                                                                   \
        if(case == 2)                                                   \
            std::cerr << "Invalid value in arguments ..." << std::endl; \
        else if(case == 1)                                              \
            std::cerr << "Invalid size arguments..." << std::endl;      \
        else                                                            \
            std::cerr << "Quick return..." << std::endl;                \
        std::cerr << "No performance data to collect." << std::endl;    \
        std::cerr << "No computations to verify." << std::endl;         \
    } while(0)

template <typename T>
constexpr double get_epsilon()
{
    using S = decltype(std::real(T{}));
    return std::numeric_limits<S>::epsilon();
}

template <typename T>
inline void rocsolver_test_check(double max_error, int tol)
{
#ifdef GOOGLE_TEST
    ASSERT_LE(max_error, tol * get_epsilon<T>());
#endif
}

inline void rocsolver_bench_output()
{
    // empty version
    std::cerr << std::endl;
}

template <typename T, typename... Ts>
inline void rocsolver_bench_output(T arg, Ts... args)
{
    using boost::format;
    format f("%|-15|");

    std::cerr << f % arg;
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

#endif
