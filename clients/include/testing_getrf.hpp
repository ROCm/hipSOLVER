/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "clientcommon.hpp"

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void getrf_initData(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Ud&                     dIpiv,
                    const int               stP,
                    Ud&                     dInfo,
                    const int               bc,
                    Th&                     hA,
                    Uh&                     hIpiv,
                    Uh&                     hInfo)
{
    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA, true);

        for(int b = 0; b < bc; ++b)
        {
            // scale A to avoid singularities
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

            // shuffle rows to test pivoting
            // always the same permuation for debugging purposes
            for(int i = 0; i < m / 2; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    tmp                        = hA[b][i + j * lda];
                    hA[b][i + j * lda]         = hA[b][m - 1 - i + j * lda];
                    hA[b][m - 1 - i + j * lda] = tmp;
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
void getrf_getError(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Td&                     dWork,
                    Ud&                     dIpiv,
                    const int               stP,
                    Ud&                     dInfo,
                    const int               bc,
                    Th&                     hA,
                    Th&                     hARes,
                    Uh&                     hIpiv,
                    Uh&                     hIpivRes,
                    Uh&                     hInfo,
                    Uh&                     hInfoRes,
                    double*                 max_err)
{
    // input data initialization
    getrf_initData<true, true, T>(
        handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_getrf(FORTRAN,
                                        false,
                                        handle,
                                        m,
                                        n,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dWork.data(),
                                        dIpiv.data(),
                                        stP,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hIpivRes.transfer_from(dIpiv));
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

        // also check pivoting (count the number of incorrect pivots)
        err = 0;
        for(int i = 0; i < min(m, n); ++i)
            if(hIpiv[b][i] != hIpivRes[b][i])
                err++;
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
void getrf_getPerfData(const hipsolverHandle_t handle,
                       const int               m,
                       const int               n,
                       Td&                     dA,
                       const int               lda,
                       const int               stA,
                       Td&                     dWork,
                       Ud&                     dIpiv,
                       const int               stP,
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
        getrf_initData<true, false, T>(
            handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us();
        for(int b = 0; b < bc; ++b)
            cblas_getrf<T>(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        *cpu_time_used = get_time_us() - *cpu_time_used;
    }

    getrf_initData<true, false, T>(
        handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        getrf_initData<false, true, T>(
            handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo);

        CHECK_ROCBLAS_ERROR(hipsolver_getrf(FORTRAN,
                                            false,
                                            handle,
                                            m,
                                            n,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dWork.data(),
                                            dIpiv.data(),
                                            stP,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    double start;
    for(int iter = 0; iter < hot_calls; iter++)
    {
        getrf_initData<false, true, T>(
            handle, m, n, dA, lda, stA, dIpiv, stP, dInfo, bc, hA, hIpiv, hInfo);

        start = get_time_us();
        hipsolver_getrf(FORTRAN,
                        false,
                        handle,
                        m,
                        n,
                        dA.data(),
                        lda,
                        stA,
                        dWork.data(),
                        dIpiv.data(),
                        stP,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us() - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool FORTRAN, bool BATCHED, bool STRIDED, typename T>
void testing_getrf(Arguments& argus)
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
    int stPRes = (argus.unit_check || argus.norm_check) ? stP : 0;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_P    = size_t(min(m, n));
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_PRes = (argus.unit_check || argus.norm_check) ? size_P : 0;

    // check invalid sizes
    bool invalid_size = (m <= 0 || n <= 0 || lda < m || bc <= 0);
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
            //                                       (rocblas_int*)nullptr,
            //                                       stP,
            //                                       (rocblas_int*)nullptr,
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
                                                  (rocblas_int*)nullptr,
                                                  stP,
                                                  (rocblas_int*)nullptr,
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
        // host_strided_batch_vector<int>   hIpivRes(size_PRes, 1, stPRes, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dIpiv.memcheck());

        // int size_W;
        // hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     getrf_getError<FORTRAN, T>(handle,
        //                                m,
        //                                n,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dWork,
        //                                dIpiv,
        //                                stP,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hARes,
        //                                hIpiv,
        //                                hIpivRes,
        //                                hInfo,
        //                                hInfoRes,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     getrf_getPerfData<FORTRAN, T>(handle,
        //                                   m,
        //                                   n,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dWork,
        //                                   dIpiv,
        //                                   stP,
        //                                   dInfo,
        //                                   bc,
        //                                   hA,
        //                                   hIpiv,
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
        host_strided_batch_vector<T>     hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<int>   hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hIpivRes(size_PRes, 1, stPRes, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<int> dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());

        int size_W;
        hipsolver_getrf_bufferSize(FORTRAN, handle, m, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            getrf_getError<FORTRAN, T>(handle,
                                       m,
                                       n,
                                       dA,
                                       lda,
                                       stA,
                                       dWork,
                                       dIpiv,
                                       stP,
                                       dInfo,
                                       bc,
                                       hA,
                                       hARes,
                                       hIpiv,
                                       hIpivRes,
                                       hInfo,
                                       hInfoRes,
                                       &max_error);

        // collect performance data
        if(argus.timing)
            getrf_getPerfData<FORTRAN, T>(handle,
                                          m,
                                          n,
                                          dA,
                                          lda,
                                          stA,
                                          dWork,
                                          dIpiv,
                                          stP,
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
        rocsolver_test_check<T>(max_error, min(m, n));

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
                rocsolver_bench_output("m", "n", "lda", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stP, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("m", "n", "lda", "strideA", "strideP", "batch_c");
                rocsolver_bench_output(m, n, lda, stA, stP, bc);
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
