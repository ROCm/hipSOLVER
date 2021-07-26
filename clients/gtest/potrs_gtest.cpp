/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potrs.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<int>> potrs_tuple;

// each A_range vector is a {N, lda, ldb};

// each B_range vector is a {nrhs, uplo};
// if uplo = 0 then upper
// if uplo = 1 then lower

// case when N = nrhs = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

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
    {1, 0},
    {1, 1},
};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_sizeA_range
//     = {{70, 70, 100}, {192, 192, 192}, {600, 700, 645}, {1000, 1000, 1000}, {1000, 2000, 2000}};
// const vector<vector<int>> large_matrix_sizeB_range = {
//     {1, 0},
//     {1, 1},
// };

Arguments potrs_setup_arguments(potrs_tuple tup)
{
    vector<int> matrix_sizeA = std::get<0>(tup);
    vector<int> matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", matrix_sizeA[0]);
    arg.set<rocblas_int>("nrhs", matrix_sizeB[0]);
    arg.set<rocblas_int>("lda", matrix_sizeA[1]);
    arg.set<rocblas_int>("ldb", matrix_sizeA[2]);

    if(matrix_sizeB[1] == 0)
        arg.set<char>("uplo", 'U');
    else
        arg.set<char>("uplo", 'L');

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class POTRS_BASE : public ::TestWithParam<potrs_tuple>
{
protected:
    POTRS_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = potrs_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<rocblas_int>("nrhs") == -1)
            testing_potrs_bad_arg<FORTRAN, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_potrs<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class POTRS : public POTRS_BASE<false>
{
};

class POTRS_FORTRAN : public POTRS_BASE<true>
{
};

// non-batch tests

TEST_P(POTRS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRS_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRS_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRS_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRS_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// batched tests

TEST_P(POTRS, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(POTRS, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(POTRS, batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(POTRS, batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

TEST_P(POTRS_FORTRAN, batched__float)
{
    run_tests<true, false, float>();
}

TEST_P(POTRS_FORTRAN, batched__double)
{
    run_tests<true, false, double>();
}

TEST_P(POTRS_FORTRAN, batched__float_complex)
{
    run_tests<true, false, rocblas_float_complex>();
}

TEST_P(POTRS_FORTRAN, batched__double_complex)
{
    run_tests<true, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          POTRS,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          POTRS_FORTRAN,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRS_FORTRAN,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
