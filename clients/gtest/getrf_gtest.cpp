/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_getrf.hpp"
#include "testing_getrf_npvt.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, int> getrf_tuple;

// each matrix_size_range vector is a {m, lda}

// for checkin_lapack tests
const vector<vector<int>> matrix_size_range = {
    // invalid
    {-1, 1},
    {20, 5},
    // normal (valid) samples
    {32, 32},
    {50, 50},
    {70, 100},
};

const vector<int> n_size_range = {
    // invalid
    -1,
    // normal (valid) samples
    16,
    20,
    40,
    100,
};

// for daily_lapack tests
const vector<vector<int>> large_matrix_size_range = {
    {192, 192},
    {640, 640},
    {1000, 1024},
};

const vector<int> large_n_size_range = {
    45,
    64,
    520,
    1024,
    2000,
};

Arguments getrf_setup_arguments(getrf_tuple tup)
{
    vector<int> matrix_size = std::get<0>(tup);
    int         n_size      = std::get<1>(tup);

    Arguments arg;

    arg.set<int>("m", matrix_size[0]);
    arg.set<int>("lda", matrix_size[1]);

    arg.set<int>("n", n_size);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <bool FORTRAN, bool NPVT>
class GETRF_BASE : public ::TestWithParam<getrf_tuple>
{
protected:
    GETRF_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = getrf_setup_arguments(GetParam());

        arg.batch_count = 1;
        if(!NPVT)
            testing_getrf<FORTRAN, BATCHED, STRIDED, T>(arg);
        else
            testing_getrf_npvt<FORTRAN, BATCHED, STRIDED, T>(arg);
    }
};

class GETRF : public GETRF_BASE<false, false>
{
};

class GETRF_FORTRAN : public GETRF_BASE<true, false>
{
};

class GETRF_NPVT : public GETRF_BASE<false, true>
{
};

class GETRF_NPVT_FORTRAN : public GETRF_BASE<true, true>
{
};

// non-batch tests
TEST_P(GETRF, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF, __float_complex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GETRF, __double_complex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GETRF_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_FORTRAN, __float_complex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GETRF_FORTRAN, __double_complex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}
TEST_P(GETRF_NPVT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_NPVT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_NPVT, __float_complex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GETRF_NPVT, __double_complex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

TEST_P(GETRF_NPVT_FORTRAN, __float)
{
    run_tests<false, false, float>();
}

TEST_P(GETRF_NPVT_FORTRAN, __double)
{
    run_tests<false, false, double>();
}

TEST_P(GETRF_NPVT_FORTRAN, __float_complex)
{
    run_tests<false, false, hipsolverComplex>();
}

TEST_P(GETRF_NPVT_FORTRAN, __double_complex)
{
    run_tests<false, false, hipsolverDoubleComplex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         GETRF_NPVT_FORTRAN,
                         Combine(ValuesIn(large_matrix_size_range), ValuesIn(large_n_size_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         GETRF_NPVT_FORTRAN,
                         Combine(ValuesIn(matrix_size_range), ValuesIn(n_size_range)));
