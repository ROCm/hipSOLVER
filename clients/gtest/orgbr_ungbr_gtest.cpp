/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgbr_ungbr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgbr_tuple;

// each size_range is a {M, N, K};

// each store_range vector is a {lda, side}
// if lda = -1, then lda < limit (invalid size)
// if lda = 0, then lda = limit
// if lda = 1, then lda > limit
// if st = 0, then side = 'L'
// if st = 1, then side = 'R'

const vector<vector<int>> store_range = {
    // always invalid
    {-1, 0},
    {-1, 1},
    // normal (valid) samples
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // always invalid
    {-1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    // invalid for side = 'L'
    {10, 30, 5},
    // invalid for side = 'R'
    {30, 10, 5},
    // always invalid
    {30, 10, 20},
    {10, 30, 20},
    // normal (valid) samples
    {30, 30, 1},
    {20, 20, 20},
    {50, 50, 50},
    {100, 100, 50}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{150, 150, 100},
//                                               {270, 270, 270},
//                                               {400, 400, 400},
//                                               {800, 800, 300},
//                                               {1000, 1000, 1000},
//                                               {1500, 1500, 800}};

Arguments orgbr_setup_arguments(orgbr_tuple tup)
{
    vector<int> size  = std::get<0>(tup);
    vector<int> store = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", size[0]);
    arg.set<rocblas_int>("n", size[1]);
    arg.set<rocblas_int>("k", size[2]);

    arg.set<rocblas_int>("lda", size[0] + store[0] * 10);
    arg.set<char>("side", store[1] == 1 ? 'R' : 'L');

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class ORGBR_UNGBR : public ::TestWithParam<orgbr_tuple>
{
protected:
    ORGBR_UNGBR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgbr_setup_arguments(GetParam());

        testing_orgbr_ungbr<FORTRAN, T>(arg);
    }
};

class ORGBR : public ORGBR_UNGBR<false>
{
};

class UNGBR : public ORGBR_UNGBR<false>
{
};

class ORGBR_FORTRAN : public ORGBR_UNGBR<true>
{
};

class UNGBR_FORTRAN : public ORGBR_UNGBR<true>
{
};

// non-batch tests

TEST_P(ORGBR, __float)
{
    run_tests<float>();
}

TEST_P(ORGBR, __double)
{
    run_tests<double>();
}

TEST_P(UNGBR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGBR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGBR_FORTRAN, __float)
{
    run_tests<float>();
}

TEST_P(ORGBR_FORTRAN, __double)
{
    run_tests<double>();
}

TEST_P(UNGBR_FORTRAN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGBR_FORTRAN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGBR,
//                          Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGBR,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGBR,
//                          Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGBR,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGBR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGBR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGBR_FORTRAN,
//                          Combine(ValuesIn(large_size_range), ValuesIn(store_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGBR_FORTRAN,
                         Combine(ValuesIn(size_range), ValuesIn(store_range)));
