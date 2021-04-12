/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_orgqr_ungqr.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> orgqr_tuple;

// each m_size_range vector is a {M, lda}

// each n_size_range vector is a {N, K}

// for checkin_lapack tests
const vector<vector<int>> m_size_range = {
    // always invalid
    {-1, 1},
    {20, 5},
    // invalid for case *
    {50, 50},
    // normal (valid) samples
    {70, 100},
    {130, 130}};

const vector<vector<int>> n_size_range = {
    // always invalid
    {-1, 1},
    {1, -1},
    {10, 20},
    // invalid for case *
    {55, 55},
    // normal (valid) samples
    {10, 10},
    {20, 20},
    {35, 25}};

// // for daily_lapack tests
// const vector<vector<int>> large_m_size_range = {{400, 410}, {640, 640}, {1000, 1024}, {2000, 2000}};

// const vector<vector<int>> large_n_size_range
//     = {{164, 162}, {198, 140}, {130, 130}, {220, 220}, {400, 200}};

Arguments orgqr_setup_arguments(orgqr_tuple tup)
{
    vector<int> m_size = std::get<0>(tup);
    vector<int> n_size = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", m_size[0]);
    arg.set<rocblas_int>("lda", m_size[1]);

    arg.set<rocblas_int>("n", n_size[0]);
    arg.set<rocblas_int>("k", n_size[1]);

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class ORGQR_UNGQR : public ::TestWithParam<orgqr_tuple>
{
protected:
    ORGQR_UNGQR() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = orgqr_setup_arguments(GetParam());

        testing_orgqr_ungqr<FORTRAN, T>(arg);
    }
};

class ORGQR : public ORGQR_UNGQR<false>
{
};

class UNGQR : public ORGQR_UNGQR<false>
{
};

class ORGQR_FORTRAN : public ORGQR_UNGQR<true>
{
};

class UNGQR_FORTRAN : public ORGQR_UNGQR<true>
{
};

// non-batch tests

TEST_P(ORGQR, __float)
{
    run_tests<float>();
}

TEST_P(ORGQR, __double)
{
    run_tests<double>();
}

TEST_P(UNGQR, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGQR, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

TEST_P(ORGQR_FORTRAN, __float)
{
    run_tests<float>();
}

TEST_P(ORGQR_FORTRAN, __double)
{
    run_tests<double>();
}

TEST_P(UNGQR_FORTRAN, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(UNGQR_FORTRAN, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGQR,
//                          Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGQR,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGQR,
//                          Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGQR,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          ORGQR_FORTRAN,
//                          Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         ORGQR_FORTRAN,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          UNGQR_FORTRAN,
//                          Combine(ValuesIn(large_m_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         UNGQR_FORTRAN,
                         Combine(ValuesIn(m_size_range), ValuesIn(n_size_range)));
