/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <bool FORTRAN, typename T, typename U>
void getrs_checkBadArgs(const hipsolverHandle_t    handle,
                        const hipsolverOperation_t trans,
                        const int                  m,
                        const int                  nrhs,
                        T                          dA,
                        const int                  lda,
                        const int                  stA,
                        U                          dIpiv,
                        const int                  stP,
                        T                          dB,
                        const int                  ldb,
                        const int                  stB,
                        U                          dInfo,
                        const int                  bc)
{
    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_getrs(FORTRAN,
                                          handle,
                                          hipsolverOperation_t(-1),
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dInfo,
                                          bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
}

template <bool FORTRAN, bool BATCHED, bool STRIDED, typename T>
void testing_getrs_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    m     = 1;
    int                    nrhs  = 1;
    int                    lda   = 1;
    int                    ldb   = 1;
    int                    stA   = 1;
    int                    stP   = 1;
    int                    stB   = 1;
    int                    bc    = 1;
    hipsolverOperation_t   trans = HIPSOLVER_OP_N;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // // check bad arguments
        // getrs_checkBadArgs<FORTRAN>(handle,
        //                             trans,
        //                             m,
        //                             nrhs,
        //                             dA.data(),
        //                             lda,
        //                             stA,
        //                             dIpiv.data(),
        //                             stP,
        //                             dB.data(),
        //                             ldb,
        //                             stB,
        //                             dInfo.data(),
        //                             bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check bad arguments
        getrs_checkBadArgs<FORTRAN>(handle,
                                    trans,
                                    m,
                                    nrhs,
                                    dA.data(),
                                    lda,
                                    stA,
                                    dIpiv.data(),
                                    stP,
                                    dB.data(),
                                    ldb,
                                    stB,
                                    dInfo.data(),
                                    bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrs_initData(const hipsolverHandle_t    handle,
                    const hipsolverOperation_t trans,
                    const int                  m,
                    const int                  nrhs,
                    Td&                        dA,
                    const int                  lda,
                    const int                  stA,
                    Ud&                        dIpiv,
                    const int                  stP,
                    Td&                        dB,
                    const int                  ldb,
                    const int                  stB,
                    const int                  bc,
                    Th&                        hA,
                    Uh&                        hIpiv,
                    Th&                        hB)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, true);

        // scale A to avoid singularities
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < m; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }

        // do the LU decomposition of matrix A w/ the reference LAPACK routine
        for(int b = 0; b < bc; ++b)
        {
            int info;
            cblas_getrf<T>(m, m, hA[b], lda, hIpiv[b], &info);
        }
    }

    if(GPU)
    {
        // now copy pivoting indices and matrices to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrs_getError(const hipsolverHandle_t    handle,
                    const hipsolverOperation_t trans,
                    const int                  m,
                    const int                  nrhs,
                    Td&                        dA,
                    const int                  lda,
                    const int                  stA,
                    Ud&                        dIpiv,
                    const int                  stP,
                    Td&                        dB,
                    const int                  ldb,
                    const int                  stB,
                    Ud&                        dInfo,
                    const int                  bc,
                    Th&                        hA,
                    Uh&                        hIpiv,
                    Th&                        hB,
                    Th&                        hBRes,
                    Uh&                        hInfo,
                    double*                    max_err)
{
    // input data initialization
    getrs_initData<true, true, T>(
        handle, trans, m, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_getrs(FORTRAN,
                                        handle,
                                        trans,
                                        m,
                                        nrhs,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dIpiv.data(),
                                        stP,
                                        dB.data(),
                                        ldb,
                                        stB,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hBRes.transfer_from(dB));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cblas_getrs<T>(trans, m, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
    }

    // error is ||hB - hBRes|| / ||hB||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using vector-induced infinity norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('I', m, nrhs, ldb, hB[b], hBRes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrs_getPerfData(const hipsolverHandle_t    handle,
                       const hipsolverOperation_t trans,
                       const int                  m,
                       const int                  nrhs,
                       Td&                        dA,
                       const int                  lda,
                       const int                  stA,
                       Ud&                        dIpiv,
                       const int                  stP,
                       Td&                        dB,
                       const int                  ldb,
                       const int                  stB,
                       Ud&                        dInfo,
                       const int                  bc,
                       Th&                        hA,
                       Uh&                        hIpiv,
                       Th&                        hB,
                       Uh&                        hInfo,
                       double*                    gpu_time_used,
                       double*                    cpu_time_used,
                       const int                  hot_calls,
                       const bool                 perf)
{
    if(!perf)
    {
        getrs_initData<true, false, T>(
            handle, trans, m, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cblas_getrs<T>(trans, m, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getrs_initData<true, false, T>(
        handle, trans, m, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getrs_initData<false, true, T>(
            handle, trans, m, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        CHECK_ROCBLAS_ERROR(hipsolver_getrs(FORTRAN,
                                            handle,
                                            trans,
                                            m,
                                            nrhs,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dIpiv.data(),
                                            stP,
                                            dB.data(),
                                            ldb,
                                            stB,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        getrs_initData<false, true, T>(
            handle, trans, m, nrhs, dA, lda, stA, dIpiv, stP, dB, ldb, stB, bc, hA, hIpiv, hB);

        start = get_time_us_sync(stream);
        hipsolver_getrs(FORTRAN,
                        handle,
                        trans,
                        m,
                        nrhs,
                        dA.data(),
                        lda,
                        stA,
                        dIpiv.data(),
                        stP,
                        dB.data(),
                        ldb,
                        stB,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool FORTRAN, bool BATCHED, bool STRIDED, typename T>
void testing_getrs(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   transC = argus.get<char>("trans");
    int                    m      = argus.get<int>("n");
    int                    nrhs   = argus.get<int>("nrhs", m);
    int                    lda    = argus.get<int>("lda", m);
    int                    ldb    = argus.get<int>("ldb", m);
    int                    stA    = argus.get<int>("strideA", lda * m);
    int                    stP    = argus.get<int>("strideP", m);
    int                    stB    = argus.get<int>("strideB", ldb * nrhs);

    hipsolverOperation_t trans     = char2hipsolver_operation(transC);
    int                  bc        = argus.batch_count;
    int                  hot_calls = argus.iters;

    int stBRes = (argus.unit_check || argus.norm_check) ? stB : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * m;
    size_t size_B    = size_t(ldb) * nrhs;
    size_t size_P    = size_t(m);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || nrhs < 0 || lda < m || ldb < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_getrs(FORTRAN,
            //                                       handle,
            //                                       trans,
            //                                       m,
            //                                       nrhs,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (int*)nullptr,
            //                                       stP,
            //                                       (T* const*)nullptr,
            //                                       ldb,
            //                                       stB,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_getrs(FORTRAN,
                                                  handle,
                                                  trans,
                                                  m,
                                                  nrhs,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (int*)nullptr,
                                                  stP,
                                                  (T*)nullptr,
                                                  ldb,
                                                  stB,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hB(size_B, 1, bc);
        // host_batch_vector<T>             hBRes(size_BRes, 1, bc);
        // host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_batch_vector<T>           dB(size_B, 1, bc);
        // device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     getrs_getError<FORTRAN, T>(handle,
        //                                trans,
        //                                m,
        //                                nrhs,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dIpiv,
        //                                stP,
        //                                dB,
        //                                ldb,
        //                                stB,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hIpiv,
        //                                hB,
        //                                hBRes,
        //                                hInfo,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     getrs_getPerfData<FORTRAN, T>(handle,
        //                                   trans,
        //                                   m,
        //                                   nrhs,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dIpiv,
        //                                   stP,
        //                                   dB,
        //                                   ldb,
        //                                   stB,
        //                                   dInfo,
        //                                   bc,
        //                                   hA,
        //                                   hIpiv,
        //                                   hB,
        //                                   hInfo,
        //                                   &gpu_time_used,
        //                                   &cpu_time_used,
        //                                   hot_calls,
        //                                   argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hB(size_B, 1, stB, bc);
        host_strided_batch_vector<T>     hBRes(size_BRes, 1, stBRes, bc);
        host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dB(size_B, 1, stB, bc);
        device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getrs_getError<FORTRAN, T>(handle,
                                       trans,
                                       m,
                                       nrhs,
                                       dA,
                                       lda,
                                       stA,
                                       dIpiv,
                                       stP,
                                       dB,
                                       ldb,
                                       stB,
                                       dInfo,
                                       bc,
                                       hA,
                                       hIpiv,
                                       hB,
                                       hBRes,
                                       hInfo,
                                       &max_error);

        // collect performance data
        if(argus.timing)
            getrs_getPerfData<FORTRAN, T>(handle,
                                          trans,
                                          m,
                                          nrhs,
                                          dA,
                                          lda,
                                          stA,
                                          dIpiv,
                                          stP,
                                          dB,
                                          ldb,
                                          stB,
                                          dInfo,
                                          bc,
                                          hA,
                                          hIpiv,
                                          hB,
                                          hInfo,
                                          &gpu_time_used,
                                          &cpu_time_used,
                                          hot_calls,
                                          argus.perf);
    }

    // validate results for rocsolver-test
    // using m * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, m);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("trans", "n", "nrhs", "lda", "ldb", "strideP", "batch_c");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output(
                    "trans", "n", "nrhs", "lda", "ldb", "strideA", "strideP", "strideB", "batch_c");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb, stA, stP, stB, bc);
            }
            else
            {
                rocsolver_bench_output("trans", "n", "nrhs", "lda", "ldb");
                rocsolver_bench_output(transC, m, nrhs, lda, ldb);
            }
            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            std::cerr << std::endl;
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}
