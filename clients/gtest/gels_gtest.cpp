/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gels.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, int, int, int> gels_params_A;

typedef std::tuple<gels_params_A, int> gels_tuple;

// each A_range tuple is a {M, N, lda, ldb};

// each B_range tuple is a {nrhs};

// case when N = nrhs = -1 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<gels_params_A> matrix_sizeA_range = {
    // invalid
    {-1, 1, 1, 1},
    {1, -1, 1, 1},
    {10, 10, 10, 1},
    {10, 10, 1, 10},
    // normal (valid) samples
    {20, 20, 20, 20},
    {30, 20, 40, 30},
    {40, 20, 40, 40},
};
const vector<int> matrix_sizeB_range = {
    // invalid
    -1,
    // normal (valid) samples
    10,
    20,
    30};

// // for daily_lapack tests
// const vector<gels_params_A> large_matrix_sizeA_range = {
//     {75, 25, 75, 75},
//     {150, 150, 150, 150},
// };
// const vector<int> large_matrix_sizeB_range = {
//     100,
//     200,
//     500,
//     1000,
// };

Arguments gels_setup_arguments(gels_tuple tup)
{
    gels_params_A matrix_sizeA = std::get<0>(tup);
    int           matrix_sizeB = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", std::get<0>(matrix_sizeA));
    arg.set<rocblas_int>("n", std::get<1>(matrix_sizeA));
    arg.set<rocblas_int>("lda", std::get<2>(matrix_sizeA));
    arg.set<rocblas_int>("ldb", std::get<3>(matrix_sizeA));
    arg.set<rocblas_int>("ldx", std::get<3>(matrix_sizeA));

    arg.set<rocblas_int>("nrhs", matrix_sizeB);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class GELS_BASE : public ::TestWithParam<gels_tuple>
{
protected:
    GELS_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gels_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<rocblas_int>("nrhs") == -1)
            testing_gels_bad_arg<FORTRAN, BATCHED, STRIDED, T>();

        arg.batch_count = 1;
        testing_gels<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class GELS : public GELS_BASE<false>
{
};

class GELS_FORTRAN : public GELS_BASE<true>
{
};

// non-batch tests

TEST_P(GELS, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GELS, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GELS, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GELS, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GELS_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GELS_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GELS_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GELS_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GELS,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GELS_FORTRAN,
//                          Combine(ValuesIn(large_matrix_sizeA_range),
//                                  ValuesIn(large_matrix_sizeB_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GELS_FORTRAN,
                         Combine(ValuesIn(matrix_sizeA_range), ValuesIn(matrix_sizeB_range)));
