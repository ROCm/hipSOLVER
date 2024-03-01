/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include "testing_sytrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> sytrf_tuple;

// each matrix_size_range vector is a {n, lda}

// each uplo_range is a {uplo}

// case when n = -1 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // invalid
    {-1, 1},
    {20, 5},
#endif
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100}};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {192, 192},
//     {640, 640},
//     {1000, 1024},
// };

Arguments sytrf_setup_arguments(sytrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    char        uplo        = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_size[0]);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    arg.set<char>("uplo", uplo);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYTRF_BASE : public ::TestWithParam<sytrf_tuple>
{
protected:
    SYTRF_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sytrf_setup_arguments(GetParam());

        if(arg.peek<char>("uplo") == 'L' && arg.peek<rocblas_int>("n") == -1)
            testing_sytrf_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_sytrf<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYTRF : public SYTRF_BASE<API_NORMAL>
{
};

class SYTRF_FORTRAN : public SYTRF_BASE<API_FORTRAN>
{
};

// non-batch tests

TEST_P(SYTRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYTRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYTRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYTRF_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRF_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(SYTRF_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(SYTRF_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRF,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYTRF_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRF_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
