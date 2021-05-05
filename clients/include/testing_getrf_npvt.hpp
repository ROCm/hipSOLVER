/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "clientcommon.hpp"

template <bool FORTRAN, typename T, typename U, typename V>
void getrf_npvt_checkBadArgs(const hipsolverHandle_t handle,
                             const int               m,
                             const int               n,
                             T                       dA,
                             const int               lda,
                             const int               stA,
                             U                       dWork,
                             V                       dIpiv,
                             const int               stP,
                             V                       dinfo,
                             const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getrf(FORTRAN, true, nullptr, m, n, dA, lda, stA, dWork, dIpiv, stP, dinfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#ifndef __HIP_PLATFORM_NVCC__
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getrf(
            FORTRAN, true, handle, m, n, (T) nullptr, lda, stA, dWork, dIpiv, stP, dinfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_getrf(
            FORTRAN, true, handle, m, n, dA, lda, stA, dWork, dIpiv, stP, (V) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <bool FORTRAN, bool BATCHED, bool STRIDED, typename T>
void testing_getrf_npvt_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    int                    m   = 1;
    int                    n   = 1;
    int                    lda = 1;
    int                    stA = 1;
    int                    stP = 1;
    int                    bc  = 1;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // getrf_npvt_checkBadArgs<FORTRAN>(
        //     handle, m, n, dA.data(), lda, stA, dWork.data(), dIpiv.data(), stP, dInfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<int> dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        getrf_npvt_checkBadArgs<FORTRAN>(
            handle, m, n, dA.data(), lda, stA, dWork.data(), dIpiv.data(), stP, dInfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_npvt_initData(const hipsolverHandle_t handle,
                         const int               m,
                         const int               n,
                         Td&                     dA,
                         const int               lda,
                         const int               stA,
                         Ud&                     dInfo,
                         const int               bc,
                         Th&                     hA,
                         Uh&                     hInfo)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        // leaving matrix as diagonal dominant so that pivoting is not required
        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < m; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_npvt_getError(const hipsolverHandle_t handle,
                         const int               m,
                         const int               n,
                         Td&                     dA,
                         const int               lda,
                         const int               stA,
                         Td&                     dWork,
                         Ud&                     dInfo,
                         const int               bc,
                         Th&                     hA,
                         Th&                     hARes,
                         Uh&                     hIpiv,
                         Uh&                     hInfo,
                         Uh&                     hInfoRes,
                         double*                 max_err)
{
    // input data initialization
    getrf_npvt_initData<true, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_getrf(FORTRAN,
                                        true,
                                        handle,
                                        m,
                                        n,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dWork.data(),
                                        (int*)nullptr,
                                        0,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);

    // expecting original matrix to be non-singular
    // error is ||hA - hARes|| / ||hA|| (ideally ||LU - Lres Ures|| / ||LU||)
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    double err;
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
    {
        err      = norm_error('F', m, n, lda, hA[b], hARes[b]);
        *max_err = err > *max_err ? err : *max_err;
    }

    // also check info for singularities
    err = 0;
    for(int b = 0; b < bc; ++b)
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    *max_err += err;
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_npvt_getPerfData(const hipsolverHandle_t handle,
                            const int               m,
                            const int               n,
                            Td&                     dA,
                            const int               lda,
                            const int               stA,
                            Td&                     dWork,
                            Ud&                     dInfo,
                            const int               bc,
                            Th&                     hA,
                            Uh&                     hIpiv,
                            Uh&                     hInfo,
                            double*                 gpu_time_used,
                            double*                 cpu_time_used,
                            const int               hot_calls,
                            const bool              perf)
{
    if(!perf)
    {
        getrf_npvt_initData<true, false, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        // cpu-lapack performance (only if no perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    getrf_npvt_initData<true, false, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getrf_npvt_initData<false, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_getrf(FORTRAN,
                                            true,
                                            handle,
                                            m,
                                            n,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dWork.data(),
                                            (int*)nullptr,
                                            0,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        getrf_npvt_initData<false, true, T>(handle, m, n, dA, lda, stA, dInfo, bc, hA, hInfo);

        start = get_time_us_sync(stream);
        hipsolver_getrf(FORTRAN,
                        true,
                        handle,
                        m,
                        n,
                        dA.data(),
                        lda,
                        stA,
                        dWork.data(),
                        (int*)nullptr,
                        0,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool FORTRAN, bool BATCHED, bool STRIDED, typename T>
void testing_getrf_npvt(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    int                    m   = argus.get<int>("m");
    int                    n   = argus.get<int>("n", m);
    int                    lda = argus.get<int>("lda", m);
    int                    stA = argus.get<int>("strideA", lda * n);
    int                    stP = argus.get<int>("strideP", min(m, n));

    int bc        = argus.batch_count;
    int hot_calls = argus.iters;

    int stARes = (argus.unit_check || argus.norm_check) ? stA : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_P    = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || lda < m || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_getrf(FORTRAN,
            //                                       true,
            //                                       handle,
            //                                       m,
            //                                       n,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (T*)nullptr,
            //                                       (int*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_getrf(FORTRAN,
                                                  true,
                                                  handle,
                                                  m,
                                                  n,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (T*)nullptr,
                                                  (int*)nullptr,
                                                  0,
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
        // host_batch_vector<T>             hARes(size_ARes, 1, bc);
        // host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     getrf_npvt_getError<FORTRAN, T>(handle,
        //                                     m,
        //                                     n,
        //                                     dA,
        //                                     lda,
        //                                     stA,
        //                                     dWork,
        //                                     dInfo,
        //                                     bc,
        //                                     hA,
        //                                     hARes,
        //                                     hIpiv,
        //                                     hInfo,
        //                                     hInfoRes,
        //                                     &max_error);

        // // collect performance data
        // if(argus.timing)
        //     getrf_npvt_getPerfData<FORTRAN, T>(handle,
        //                                        m,
        //                                        n,
        //                                        dA,
        //                                        lda,
        //                                        stA,
        //                                        dWork,
        //                                        dInfo,
        //                                        bc,
        //                                        hA,
        //                                        hIpiv,
        //                                        hInfo,
        //                                        &gpu_time_used,
        //                                        &cpu_time_used,
        //                                        hot_calls,
        //                                        argus.perf);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>     hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>     hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getrf_npvt_getError<FORTRAN, T>(handle,
                                            m,
                                            n,
                                            dA,
                                            lda,
                                            stA,
                                            dWork,
                                            dInfo,
                                            bc,
                                            hA,
                                            hARes,
                                            hIpiv,
                                            hInfo,
                                            hInfoRes,
                                            &max_error);

        // collect performance data
        if(argus.timing)
            getrf_npvt_getPerfData<FORTRAN, T>(handle,
                                               m,
                                               n,
                                               dA,
                                               lda,
                                               stA,
                                               dWork,
                                               dInfo,
                                               bc,
                                               hA,
                                               hIpiv,
                                               hInfo,
                                               &gpu_time_used,
                                               &cpu_time_used,
                                               hot_calls,
                                               argus.perf);
    }

    // validate results for rocsolver-test
    // using min(m,n) * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, min(m, n));

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
                rocsolver_bench_output("m", "n", "lda", "batch_c");
                rocsolver_bench_output(m, n, lda, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, bc);
            }
            else
            {
                rocsolver_bench_output("m", "n", "lda");
                rocsolver_bench_output(m, n, lda);
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
