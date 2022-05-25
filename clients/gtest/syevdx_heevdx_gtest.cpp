/* ************************************************************************
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syevdx_heevdx.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> syevdx_heevdx_tuple;

// each size_range vector is a {n, lda, vl, vu, il, iu}

// each op_range vector is a {jobz, range, uplo}

// case when n == 1, jobz == N, range == V, uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range
    = {{'N', 'V', 'L'}, {'V', 'A', 'U'}, {'V', 'V', 'L'}, {'V', 'I', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1, 0, 10, 1, 1},
    {10, 10, 0, 10, 1, 1},
    // normal (valid) samples
    {1, 1, 0, 10, 1, 1},
    {12, 12, -20, 20, 10, 12},
    {20, 30, 5, 15, 1, 20},
    {35, 35, -10, 10, 1, 15},
    {50, 60, -15, -5, 20, 30}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range
//     = {{192, 192, 5, 15, 100, 170}, {256, 270, -10, 10, 1, 256}, {300, 300, -15, -5, 200, 300}};

template <typename T>
Arguments syevdx_heevdx_setup_arguments(syevdx_heevdx_tuple tup)
{
    using S = decltype(std::real(T{}));

    vector<int>  size = std::get<0>(tup);
    vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);
    arg.set<double>("vl", size[2]);
    arg.set<double>("vu", size[3]);
    arg.set<rocblas_int>("il", size[4]);
    arg.set<rocblas_int>("iu", size[5]);

    arg.set<char>("jobz", op[0]);
    arg.set<char>("range", op[1]);
    arg.set<char>("uplo", op[2]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class SYEVDX_HEEVDX : public ::TestWithParam<syevdx_heevdx_tuple>
{
protected:
    SYEVDX_HEEVDX() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syevdx_heevdx_setup_arguments<T>(GetParam());

        if(arg.peek<rocblas_int>("n") == 1 && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("range") == 'V' && arg.peek<char>("uplo") == 'L')
            testing_syevdx_heevdx_bad_arg<FORTRAN, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_syevdx_heevdx<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class SYEVDX : public SYEVDX_HEEVDX<false>
{
};

class HEEVDX : public SYEVDX_HEEVDX<false>
{
};

class SYEVDX_FORTRAN : public SYEVDX_HEEVDX<true>
{
};

class HEEVDX_FORTRAN : public SYEVDX_HEEVDX<true>
{
};

// non-batch tests

TEST_P(SYEVDX, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVDX, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVDX, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVDX, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYEVDX_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVDX_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVDX_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVDX_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVDX,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, SYEVDX, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVDX,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, HEEVDX, Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVDX_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVDX_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVDX_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVDX_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
