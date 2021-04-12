/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_gebrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> gebrd_tuple;

// each matrix_size_range is a {m, lda}

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {50, 50},
    {70, 100},
    {130, 130},
    {150, 200}};

const vector<int> n_size_range = {
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    120,
    150};

// // for daily_lapack tests
// const vector<vector<int>> large_matrix_size_range = {
//     {152, 152},
//     {640, 640},
//     {1000, 1024},
// };

// const vector<int> large_n_size_range = {64, 98, 130, 220};

Arguments gebrd_setup_arguments(gebrd_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int         n_size      = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("m", matrix_size[0]);
    arg.set<rocblas_int>("n", n_size);
    arg.set<rocblas_int>("lda", matrix_size[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN>
class GEBRD_BASE : public ::TestWithParam<gebrd_tuple>
{
protected:
    GEBRD_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = gebrd_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_gebrd<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class GEBRD : public GEBRD_BASE<false>
{
};

class GEBRD_FORTRAN : public GEBRD_BASE<true>
{
};

// non-batch tests

TEST_P(GEBRD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBRD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBRD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBRD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(GEBRD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GEBRD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GEBRD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(GEBRD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GEBRD,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          GEBRD_FORTRAN,
//                          Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GEBRD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
