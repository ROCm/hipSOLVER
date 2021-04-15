/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gesvd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> gesvd_tuple;

// each size_range vector is a {m, n, fa};
// if fa = 0 then no fast algorithm is allowed
// if fa = 1 fast algorithm is used when possible

// each opt_range vector is a {lda, ldu, ldv, leftsv, rightsv};
// if ldx = -1 then ldx < limit (invalid size)
// if ldx = 0 then ldx = limit
// if ldx = 1 then ldx > limit
// if leftsv (rightsv) = 0 then overwrite singular vectors
// if leftsv (rightsv) = 1 then compute singular vectors
// if leftsv (rightsv) = 2 then compute all orthogonal matrix
// if leftsv (rightsv) = 3 then no singular vectors are computed

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // invalid
    {-1, 1, 0},
    {1, -1, 0},
    // normal (valid) samples
    {1, 1, 0},
    {20, 20, 0},
    {40, 30, 0},
    {60, 30, 0},
    {60, 30, 1}};

const vector<vector<int>> opt_range = {
    // invalid
    {-1, 0, 0, 2, 2},
    {0, -1, 0, 1, 2},
    {0, 0, -1, 2, 1},
    {0, 0, 0, 0, 0},
    // normal (valid) samples
    {1, 1, 1, 3, 3},
    {0, 0, 1, 3, 2},
    {0, 1, 0, 3, 1},
    {0, 1, 1, 3, 0},
    {1, 0, 0, 2, 3},
    {1, 0, 1, 2, 2},
    {1, 1, 0, 2, 1},
    {0, 0, 0, 2, 0},
    {0, 0, 0, 1, 3},
    {0, 0, 0, 1, 2},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 1, 0},
    {0, 0, 0, 0, 3},
    {0, 0, 0, 0, 2},
    {0, 0, 0, 0, 1}};

// for daily_lapack tests
const vector<vector<int>> large_size_range = {{120, 100, 0}, {300, 120, 0}, {300, 120, 1}};

const vector<vector<int>> large_opt_range = {{0, 0, 0, 3, 3},
                                             {1, 0, 0, 0, 1},
                                             {0, 1, 0, 1, 0},
                                             {0, 0, 1, 1, 1},
                                             {0, 0, 0, 3, 0},
                                             {0, 0, 0, 1, 3},
                                             {0, 0, 0, 3, 2}};

Arguments gesvd_setup_arguments(gesvd_tuple tup)
{
    vector<int> size = std::get<0>(tup);
    vector<int> opt  = std::get<1>(tup);

    Arguments arg;

    // sizes
    rocblas_int m = size[0];
    rocblas_int n = size[1];
    arg.set<rocblas_int>("m", m);
    arg.set<rocblas_int>("n", n);

    // // fast algorithm
    // if(size[2] == 0)
    //     arg.set<char>("fast_alg", 'I');
    // else
    //     arg.set<char>("fast_alg", 'O');

    // leading dimensions
    arg.set<rocblas_int>("lda", m + opt[0] * 10);
    arg.set<rocblas_int>("ldu", m + opt[1] * 10);
    if(opt[4] == 2)
        arg.set<rocblas_int>("ldv", n + opt[2] * 10);
    else
        arg.set<rocblas_int>("ldv", min(m, n) + opt[2] * 10);

    // vector options
    if(opt[3] == 0)
        arg.set<char>("jobu", 'O');
    else if(opt[3] == 1)
        arg.set<char>("jobu", 'S');
    else if(opt[3] == 2)
        arg.set<char>("jobu", 'A');
    else
        arg.set<char>("jobu", 'N');

    if(opt[4] == 0)
        arg.set<char>("jobv", 'O');
    else if(opt[4] == 1)
        arg.set<char>("jobv", 'S');
    else if(opt[4] == 2)
        arg.set<char>("jobv", 'A');
    else
        arg.set<char>("jobv", 'N');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class GESVD_BASE : public ::TestWithParam<gesvd_tuple>
{
protected:
    GESVD_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gesvd_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_gesvd<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class GESVD : public GESVD_BASE<false>
{
};

class GESVD_FORTRAN : public GESVD_BASE<true>
{
};

// non-batch tests

TEST_P(GESVD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GESVD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GESVD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GESVD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GESVD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESVD,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack, GESVD, Combine(ValuesIn(size_range), ValuesIn(opt_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GESVD_FORTRAN,
                         Combine(ValuesIn(large_size_range), ValuesIn(large_opt_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GESVD_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(opt_range)));
