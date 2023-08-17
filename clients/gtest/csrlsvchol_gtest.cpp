/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "testing_csrlsvchol.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

typedef std::tuple<int, vector<int>> csrlsvchol_tuple;

// each n_range vector is {n}

// each nnz_range vector is {nnzA, reorder}

// case when n = 20 and nnz = 60 also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<int> n_range = {
    20,
    50,
};
const vector<vector<int>> nnz_range = {
    {60, 0},
    {60, 1},
    {100, 2},
    {140, 3},
};

// for daily_lapack tests
const vector<int> large_n_range = {
    // normal (valid) samples
    100,
    250,
};
const vector<vector<int>> large_nnz_range = {
    // normal (valid) samples
    {300, 0},
    {300, 1},
    {500, 2},
    {700, 3},
};

Arguments csrlsvchol_setup_arguments(csrlsvchol_tuple tup)
{
    int         n_v   = std::get<0>(tup);
    vector<int> nnz_v = std::get<1>(tup);

    Arguments arg;

    arg.set<rocblas_int>("n", n_v);
    arg.set<rocblas_int>("nnzA", nnz_v[0]);
    arg.set<rocblas_int>("reorder", nnz_v[1]);

    arg.timing = 0;

    return arg;
}

template <bool HOST>
class CSRLSVCHOL_BASE : public ::TestWithParam<csrlsvchol_tuple>
{
protected:
    CSRLSVCHOL_BASE() {}
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    void run_tests()
    {
        Arguments arg = csrlsvchol_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 20 && arg.peek<rocblas_int>("nnzA") == 60)
            testing_csrlsvchol_bad_arg<HOST, T>();

        arg.batch_count = 1;
        testing_csrlsvchol<HOST, T>(arg);
    }
};

class CSRLSVCHOL : public CSRLSVCHOL_BASE<false>
{
};

class CSRLSVCHOLHOST : public CSRLSVCHOL_BASE<true>
{
};

// non-batch tests

TEST_P(CSRLSVCHOL, __float)
{
    run_tests<float>();
}

TEST_P(CSRLSVCHOL, __double)
{
    run_tests<double>();
}

// TEST_P(CSRLSVCHOL, __float_complex)
// {
//     run_tests<rocblas_float_complex>();
// }

// TEST_P(CSRLSVCHOL, __double_complex)
// {
//     run_tests<rocblas_double_complex>();
// }

TEST_P(CSRLSVCHOLHOST, __float)
{
    run_tests<float>();
}

TEST_P(CSRLSVCHOLHOST, __double)
{
    run_tests<double>();
}

// TEST_P(CSRLSVCHOLHOST, __float_complex)
// {
//     run_tests<rocblas_float_complex>();
// }

// TEST_P(CSRLSVCHOLHOST, __double_complex)
// {
//     run_tests<rocblas_double_complex>();
// }

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRLSVCHOL,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRLSVCHOL,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range)));

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         CSRLSVCHOLHOST,
                         Combine(ValuesIn(large_n_range), ValuesIn(large_nnz_range)));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         CSRLSVCHOLHOST,
                         Combine(ValuesIn(n_range), ValuesIn(nnz_range)));
