/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void orgqr_ungqr_initData(const hipsolverHandle_t handle,
                          const int               m,
                          const int               n,
                          const int               k,
                          Td&                     dA,
                          const int               lda,
                          Td&                     dIpiv,
                          Th&                     hA,
                          Th&                     hIpiv,
                          std::vector<T>&         hW,
                          size_t                  size_W)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);

        // scale to avoid singularities
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < k; ++j)
            {
                if(i == j)
                    hA[0][i + j * lda] += 400;
                else
                    hA[0][i + j * lda] -= 4;
            }
        }

        // compute QR factorization
        cblas_geqrf<T>(m, n, hA[0], lda, hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
    }
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgqr_ungqr_getError(const hipsolverHandle_t handle,
                          const int               m,
                          const int               n,
                          const int               k,
                          Td&                     dA,
                          const int               lda,
                          Td&                     dIpiv,
                          Td&                     dWork,
                          const int               lwork,
                          Ud&                     dInfo,
                          Th&                     hA,
                          Th&                     hAr,
                          Th&                     hIpiv,
                          Uh&                     hInfo,
                          double*                 max_err)
{
    size_t         size_W = size_t(n);
    std::vector<T> hW(size_W);

    // initialize data
    orgqr_ungqr_initData<true, true, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_orgqr_ungqr(
        FORTRAN, handle, m, n, k, dA.data(), lda, dIpiv.data(), dWork.data(), lwork, dInfo.data()));
    CHECK_HIP_ERROR(hAr.transfer_from(dA));

    // CPU lapack
    cblas_orgqr_ungqr<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);

    // error is ||hA - hAr|| / ||hA||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, lda, hA[0], hAr[0]);
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void orgqr_ungqr_getPerfData(const hipsolverHandle_t handle,
                             const int               m,
                             const int               n,
                             const int               k,
                             Td&                     dA,
                             const int               lda,
                             Td&                     dIpiv,
                             Td&                     dWork,
                             const int               lwork,
                             Ud&                     dInfo,
                             Th&                     hA,
                             Th&                     hIpiv,
                             Uh&                     hInfo,
                             double*                 gpu_time_used,
                             double*                 cpu_time_used,
                             const int               hot_calls,
                             const bool              perf)
{
    size_t         size_W = size_t(n);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        orgqr_ungqr_initData<true, false, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_orgqr_ungqr<T>(m, n, k, hA[0], lda, hIpiv[0], hW.data(), size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    orgqr_ungqr_initData<true, false, T>(handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        orgqr_ungqr_initData<false, true, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_orgqr_ungqr(FORTRAN,
                                                  handle,
                                                  m,
                                                  n,
                                                  k,
                                                  dA.data(),
                                                  lda,
                                                  dIpiv.data(),
                                                  dWork.data(),
                                                  lwork,
                                                  dInfo.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        orgqr_ungqr_initData<false, true, T>(
            handle, m, n, k, dA, lda, dIpiv, hA, hIpiv, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_orgqr_ungqr(FORTRAN,
                              handle,
                              m,
                              n,
                              k,
                              dA.data(),
                              lda,
                              dIpiv.data(),
                              dWork.data(),
                              lwork,
                              dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool FORTRAN, typename T>
void testing_orgqr_ungqr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    int                    n   = argus.get<int>("n");
    int                    m   = argus.get<int>("m", n);
    int                    k   = argus.get<int>("k", n);
    int                    lda = argus.get<int>("lda", m);

    int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_P    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Ar = (argus.unit_check || argus.norm_check) ? size_A : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || k < 0 || lda < m || n > m || k > n);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_orgqr_ungqr(FORTRAN,
                                                    handle,
                                                    m,
                                                    n,
                                                    k,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T>     hAr(size_Ar, 1, size_Ar, 1);
    host_strided_batch_vector<T>     hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T>   dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_orgqr_ungqr_bufferSize(
        FORTRAN, handle, m, n, k, dA.data(), lda, dIpiv.data(), &size_W);
    device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
        orgqr_ungqr_getError<FORTRAN, T>(handle,
                                         m,
                                         n,
                                         k,
                                         dA,
                                         lda,
                                         dIpiv,
                                         dWork,
                                         size_W,
                                         dInfo,
                                         hA,
                                         hAr,
                                         hIpiv,
                                         hInfo,
                                         &max_error);

    // collect performance data
    if(argus.timing)
        orgqr_ungqr_getPerfData<FORTRAN, T>(handle,
                                            m,
                                            n,
                                            k,
                                            dA,
                                            lda,
                                            dIpiv,
                                            dWork,
                                            size_W,
                                            dInfo,
                                            hA,
                                            hIpiv,
                                            hInfo,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf);

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
            rocsolver_bench_output("m", "n", "k", "lda");
            rocsolver_bench_output(m, n, k, lda);

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
