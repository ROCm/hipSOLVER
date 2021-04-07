/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_sytrd_hetrd.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, char> sytrd_tuple;

// each matrix_size_range is a {n, lda}

const vector<char> uplo_range = {'L', 'U'};

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

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {152, 152},
    {640, 640},
    {1000, 1024},
};

Arguments sytrd_setup_arguments(sytrd_tuple tup)
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
class SYTRD_HETRD : public ::TestWithParam<sytrd_tuple>
{
protected:
    SYTRD_HETRD() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = sytrd_setup_arguments(GetParam());

        arg.batch_count = 1;
        testing_sytrd_hetrd<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class SYTRD : public SYTRD_HETRD<false>
{
};

class HETRD : public SYTRD_HETRD<false>
{
};

class SYTRD_FORTRAN : public SYTRD_HETRD<true>
{
};

class HETRD_FORTRAN : public SYTRD_HETRD<true>
{
};

// non-batch tests

TEST_P(SYTRD, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRD, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HETRD, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HETRD, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

TEST_P(SYTRD_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYTRD_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HETRD_FORTRAN, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HETRD_FORTRAN, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HETRD,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         SYTRD_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYTRD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         HETRD_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(uplo_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HETRD_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(uplo_range)));
