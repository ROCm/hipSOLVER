/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_potrf.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> potrf_tuple;

// each size_range vector is a {N, lda}
// if singular = 1, then the used matrix for the tests is not positive definite

// each uplo_range is a {uplo}

// case when n = 0 and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<char> uplo_range = {'L', 'U'};

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {10, 2},
    // normal (valid) samples
    {10, 10},
    {20, 30},
    {50, 50},
    {70, 80}};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192},
    {640, 960},
    {1000, 1000},
    {1024, 1024},
    {2000, 2000},
};

Arguments potrf_setup_arguments(potrf_tuple tup)
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

template <bool FORTRAN>
class POTRF_BASE : public ::TestWithParam<potrf_tuple>
{
protected:
    POTRF_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = potrf_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_potrf<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class POTRF : public POTRF_BASE<false>
{
};

class POTRF_FORTRAN : public POTRF_BASE<true>
{
};

// non-batch tests
TEST_P(POTRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(POTRF_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(POTRF_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(POTRF_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(POTRF_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         POTRF_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         POTRF_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
