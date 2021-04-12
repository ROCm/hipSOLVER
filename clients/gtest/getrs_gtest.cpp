/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> getrs_tuple;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, trans};
// if trans = 0 then no transpose
// if trans = 1 then transpose
// if trans = 2 then conjugate transpose

// for checkin_lapack tests
const vector<vector<int>> matrix_sizeA_range = {
    // invalid
    {-1, 1, 1},
    {10, 2, 10},
    {10, 10, 2},
    /// normal (valid) samples
    {20, 20, 20},
    {30, 50, 30},
    {30, 30, 50},
    {50, 60, 60}};

const vector<vector<int>> matrix_sizeB_range = {
    // invalid
    {-1, 0},
    // normal (valid) samples
    {10, 0},
    {20, 1},
    {30, 2},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_sizeA_range
//     = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};

// const vector<vector<int>> large_matrix_sizeB_range = {
//     {100, 0},
//     {150, 0},
//     {200, 1},
//     {524, 2},
//     {1000, 2},
// };

Arguments getrs_setup_arguments(getrs_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB[0]);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    if(matrix_sizeB[1] == 0)
        arg.set<char>("trans", 'N');
    else if(matrix_sizeB[1] == 1)
        arg.set<char>("trans", 'T');
    else
        arg.set<char>("trans", 'C');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class GETRS_BASE : public ::TestWithParam<getrs_tuple>
{
protected:
    GETRS_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrs_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_getrs<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class GETRS : public GETRS_BASE<false>
{
};

class GETRS_FORTRAN : public GETRS_BASE<true>
{
};

// non-batch tests

TEST_P(GETRS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GETRS_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRS_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRS_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GETRS_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GETRS_FORTRAN,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRS_FORTRAN,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
