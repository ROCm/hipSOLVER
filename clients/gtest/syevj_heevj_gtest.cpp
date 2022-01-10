/* ************************************************************************
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#include "testing_syevj_heevj.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<vector<int>, vector<char>> syevj_heevj_tuple;

// each size_range vector is a {n, lda}

// each op_range vector is a {jobz, uplo}

// case when n == -1, jobz == N, and uplo = L will also execute the bad arguments test
// (null handle, null pointers and invalid values)

const vector<vector<char>> op_range = {{'N', 'L'}, {'N', 'U'}, {'V', 'L'}, {'V', 'U'}};

// for checkin_lapack tests
const vector<vector<int>> size_range = {
    // normal (valid) samples
    {1, 1},
    {12, 12},
    {20, 30},
    {35, 35},
    {50, 60}};

// // for daily_lapack tests
// const vector<vector<int>> large_size_range = {{192, 192}, {256, 270}, {300, 300}};

Arguments syevj_heevj_setup_arguments(syevj_heevj_tuple tup)
{
    vector<int>  size = std::get<0>(tup);
    vector<char> op   = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", size[0]);
    arg.set<rocblas_int>("lda", size[1]);

    arg.set<char>("jobz", op[0]);
    arg.set<char>("uplo", op[1]);

    // only testing standard use case/defaults for strides

    arg.timing = 0;

    return arg;
}

template <testAPI_t API>
class SYEVJ_HEEVJ : public ::TestWithParam<syevj_heevj_tuple>
{
protected:
    SYEVJ_HEEVJ() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <bool BATCHED, bool STRIDED, typename T>
    void run_tests()
    {
        Arguments arg = syevj_heevj_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == -1 && arg.peek<char>("jobz") == 'N'
           && arg.peek<char>("uplo") == 'L')
            testing_syevj_heevj_bad_arg<API, BATCHED, STRIDED, T>();

        arg.batch_count = (BATCHED || STRIDED ? 3 : 1);
        testing_syevj_heevj<API, BATCHED, STRIDED, T>(arg);
    }
};

class SYEVJ_COMPAT : public SYEVJ_HEEVJ<API_COMPAT>
{
};

class HEEVJ_COMPAT : public SYEVJ_HEEVJ<API_COMPAT>
{
};

// non-batch tests

TEST_P(SYEVJ_COMPAT, __float)
{
    run_tests<false, false, float>();
}

TEST_P(SYEVJ_COMPAT, __double)
{
    run_tests<false, false, double>();
}

TEST_P(HEEVJ_COMPAT, __float_complex)
{
    run_tests<false, false, rocblas_float_complex>();
}

TEST_P(HEEVJ_COMPAT, __double_complex)
{
    run_tests<false, false, rocblas_double_complex>();
}

// strided_batched tests

TEST_P(SYEVJ_COMPAT, strided_batched__float)
{
    run_tests<false, true, float>();
}

TEST_P(SYEVJ_COMPAT, strided_batched__double)
{
    run_tests<false, true, double>();
}

TEST_P(HEEVJ_COMPAT, strided_batched__float_complex)
{
    run_tests<false, true, rocblas_float_complex>();
}

TEST_P(HEEVJ_COMPAT, strided_batched__double_complex)
{
    run_tests<false, true, rocblas_double_complex>();
}

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          SYEVJ_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         SYEVJ_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));

// INSTANTIATE_TEST_SUITE_P(daily_lapack,
//                          HEEVJ_COMPAT,
//                          Combine(ValuesIn(large_size_range), ValuesIn(op_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         HEEVJ_COMPAT,
                         Combine(ValuesIn(size_range), ValuesIn(op_range)));
