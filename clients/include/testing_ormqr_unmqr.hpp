/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <bool FORTRAN, bool COMPLEX, typename T, typename U>
void ormqr_unmqr_checkBadArgs(const hipsolverHandle_t    handle,
                              const hipsolverSideMode_t  side,
                              const hipsolverOperation_t trans,
                              const int                  m,
                              const int                  n,
                              const int                  k,
                              T                          dA,
                              const int                  lda,
                              T                          dIpiv,
                              T                          dC,
                              const int                  ldc,
                              T                          dWork,
                              const int                  lwork,
                              U                          dInfo)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_ormqr_unmqr(
            FORTRAN, nullptr, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, dWork, lwork, dInfo),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                handle,
                                                hipsolverSideMode_t(-1),
                                                trans,
                                                m,
                                                n,
                                                k,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                handle,
                                                side,
                                                hipsolverOperation_t(-1),
                                                m,
                                                n,
                                                k,
                                                dA,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    if(COMPLEX)
        EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                    handle,
                                                    side,
                                                    HIPSOLVER_OP_T,
                                                    m,
                                                    n,
                                                    k,
                                                    dA,
                                                    lda,
                                                    dIpiv,
                                                    dC,
                                                    ldc,
                                                    dWork,
                                                    lwork,
                                                    dInfo),
                              HIPSOLVER_STATUS_INVALID_VALUE);
    else
        EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                    handle,
                                                    side,
                                                    HIPSOLVER_OP_C,
                                                    m,
                                                    n,
                                                    k,
                                                    dA,
                                                    lda,
                                                    dIpiv,
                                                    dC,
                                                    ldc,
                                                    dWork,
                                                    lwork,
                                                    dInfo),
                              HIPSOLVER_STATUS_INVALID_VALUE);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                handle,
                                                side,
                                                trans,
                                                m,
                                                n,
                                                k,
                                                (T) nullptr,
                                                lda,
                                                dIpiv,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                handle,
                                                side,
                                                trans,
                                                m,
                                                n,
                                                k,
                                                dA,
                                                lda,
                                                (T) nullptr,
                                                dC,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                handle,
                                                side,
                                                trans,
                                                m,
                                                n,
                                                k,
                                                dA,
                                                lda,
                                                dIpiv,
                                                (T) nullptr,
                                                ldc,
                                                dWork,
                                                lwork,
                                                dInfo),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <bool FORTRAN, typename T, bool COMPLEX = is_complex<T>>
void testing_ormqr_unmqr_bad_arg()
{
    // safe arguments
    hipsolver_local_handle handle;
    hipsolverSideMode_t    side  = HIPSOLVER_SIDE_LEFT;
    hipsolverOperation_t   trans = HIPSOLVER_OP_N;
    int                    k     = 1;
    int                    m     = 1;
    int                    n     = 1;
    int                    lda   = 1;
    int                    ldc   = 1;

    // memory allocation
    device_strided_batch_vector<T>   dA(1, 1, 1, 1);
    device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(1, 1, 1, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dIpiv.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_ormqr_unmqr_bufferSize(FORTRAN,
                                     handle,
                                     side,
                                     trans,
                                     m,
                                     n,
                                     k,
                                     dA.data(),
                                     lda,
                                     dIpiv.data(),
                                     dC.data(),
                                     ldc,
                                     &size_W);
    device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check bad arguments
    ormqr_unmqr_checkBadArgs<FORTRAN, COMPLEX>(handle,
                                               side,
                                               trans,
                                               m,
                                               n,
                                               k,
                                               dA.data(),
                                               lda,
                                               dIpiv.data(),
                                               dC.data(),
                                               ldc,
                                               dWork.data(),
                                               size_W,
                                               dInfo.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void ormqr_unmqr_initData(const hipsolverHandle_t    handle,
                          const hipsolverSideMode_t  side,
                          const hipsolverOperation_t trans,
                          const int                  m,
                          const int                  n,
                          const int                  k,
                          Td&                        dA,
                          const int                  lda,
                          Td&                        dIpiv,
                          Td&                        dC,
                          const int                  ldc,
                          Th&                        hA,
                          Th&                        hIpiv,
                          Th&                        hC,
                          std::vector<T>&            hW,
                          size_t                     size_W)
{
    if(CPU)
    {
        int nq = (side == HIPSOLVER_SIDE_LEFT) ? m : n;

        rocblas_init<T>(hA, true);
        rocblas_init<T>(hIpiv, true);
        rocblas_init<T>(hC, true);

        // scale to avoid singularities
        for(int i = 0; i < nq; ++i)
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
        cblas_geqrf<T>(nq, k, hA[0], lda, hIpiv[0], hW.data(), size_W);
    }

    if(GPU)
    {
        // copy data from CPU to device
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dIpiv.transfer_from(hIpiv));
        CHECK_HIP_ERROR(dC.transfer_from(hC));
    }
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void ormqr_unmqr_getError(const hipsolverHandle_t    handle,
                          const hipsolverSideMode_t  side,
                          const hipsolverOperation_t trans,
                          const int                  m,
                          const int                  n,
                          const int                  k,
                          Td&                        dA,
                          const int                  lda,
                          Td&                        dIpiv,
                          Td&                        dC,
                          const int                  ldc,
                          Td&                        dWork,
                          const int                  lwork,
                          Ud&                        dInfo,
                          Th&                        hA,
                          Th&                        hIpiv,
                          Th&                        hC,
                          Th&                        hCr,
                          Uh&                        hInfo,
                          double*                    max_err)
{
    size_t         size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    // initialize data
    ormqr_unmqr_initData<true, true, T>(
        handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_ormqr_unmqr(FORTRAN,
                                              handle,
                                              side,
                                              trans,
                                              m,
                                              n,
                                              k,
                                              dA.data(),
                                              lda,
                                              dIpiv.data(),
                                              dC.data(),
                                              ldc,
                                              dWork.data(),
                                              lwork,
                                              dInfo.data()));
    CHECK_HIP_ERROR(hCr.transfer_from(dC));

    // CPU lapack
    cblas_ormqr_unmqr<T>(side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W);

    // error is ||hC - hCr|| / ||hC||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    *max_err = norm_error('F', m, n, ldc, hC[0], hCr[0]);
}

template <bool FORTRAN, typename T, typename Td, typename Ud, typename Th, typename Uh>
void ormqr_unmqr_getPerfData(const hipsolverHandle_t    handle,
                             const hipsolverSideMode_t  side,
                             const hipsolverOperation_t trans,
                             const int                  m,
                             const int                  n,
                             const int                  k,
                             Td&                        dA,
                             const int                  lda,
                             Td&                        dIpiv,
                             Td&                        dC,
                             const int                  ldc,
                             Td&                        dWork,
                             const int                  lwork,
                             Ud&                        dInfo,
                             Th&                        hA,
                             Th&                        hIpiv,
                             Th&                        hC,
                             Uh&                        hInfo,
                             double*                    gpu_time_used,
                             double*                    cpu_time_used,
                             const int                  hot_calls,
                             const bool                 perf)
{
    size_t         size_W = max(max(m, n), k);
    std::vector<T> hW(size_W);

    if(!perf)
    {
        ormqr_unmqr_initData<true, false, T>(
            handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        cblas_ormqr_unmqr<T>(
            side, trans, m, n, k, hA[0], lda, hIpiv[0], hC[0], ldc, hW.data(), size_W);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    ormqr_unmqr_initData<true, false, T>(
        handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ormqr_unmqr_initData<false, true, T>(
            handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        CHECK_ROCBLAS_ERROR(hipsolver_ormqr_unmqr(FORTRAN,
                                                  handle,
                                                  side,
                                                  trans,
                                                  m,
                                                  n,
                                                  k,
                                                  dA.data(),
                                                  lda,
                                                  dIpiv.data(),
                                                  dC.data(),
                                                  ldc,
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
        ormqr_unmqr_initData<false, true, T>(
            handle, side, trans, m, n, k, dA, lda, dIpiv, dC, ldc, hA, hIpiv, hC, hW, size_W);

        start = get_time_us_sync(stream);
        hipsolver_ormqr_unmqr(FORTRAN,
                              handle,
                              side,
                              trans,
                              m,
                              n,
                              k,
                              dA.data(),
                              lda,
                              dIpiv.data(),
                              dC.data(),
                              ldc,
                              dWork.data(),
                              lwork,
                              dInfo.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool FORTRAN, typename T, bool COMPLEX = is_complex<T>>
void testing_ormqr_unmqr(Arguments& argus)
{
    // get arguments
    hipsolver_local_handle handle;
    char                   sideC  = argus.get<char>("side");
    char                   transC = argus.get<char>("trans");
    int                    m, n, k;
    if(sideC == 'L')
    {
        m = argus.get<int>("m");
        n = argus.get<int>("n", m);
        k = argus.get<int>("k", m);
    }
    else
    {
        n = argus.get<int>("n");
        m = argus.get<int>("m", n);
        k = argus.get<int>("k", n);
    }
    int lda = argus.get<int>("lda", sideC == 'L' ? m : n);
    int ldc = argus.get<int>("ldc", m);

    hipsolverSideMode_t  side      = char2hipsolver_side(sideC);
    hipsolverOperation_t trans     = char2hipsolver_operation(transC);
    int                  hot_calls = argus.iters;

    // check non-supported values
    bool invalid_value
        = ((COMPLEX && trans == HIPSOLVER_OP_T) || (!COMPLEX && trans == HIPSOLVER_OP_C));
    if(invalid_value)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                    handle,
                                                    side,
                                                    trans,
                                                    m,
                                                    n,
                                                    k,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    bool   left      = (side == HIPSOLVER_SIDE_LEFT);
    size_t size_A    = size_t(lda) * k;
    size_t size_P    = size_t(k);
    size_t size_C    = size_t(ldc) * n;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_Cr = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = ((m < 0 || n < 0 || k < 0 || ldc < m) || (left && (lda < m || k > m))
                         || (!left && (lda < n || k > n)));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_ormqr_unmqr(FORTRAN,
                                                    handle,
                                                    side,
                                                    trans,
                                                    m,
                                                    n,
                                                    k,
                                                    (T*)nullptr,
                                                    lda,
                                                    (T*)nullptr,
                                                    (T*)nullptr,
                                                    ldc,
                                                    (T*)nullptr,
                                                    0,
                                                    (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory allocations
    host_strided_batch_vector<T>     hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T>     hCr(size_Cr, 1, size_Cr, 1);
    host_strided_batch_vector<T>     hIpiv(size_P, 1, size_P, 1);
    host_strided_batch_vector<T>     hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<int>   hInfo(1, 1, 1, 1);
    device_strided_batch_vector<T>   dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T>   dIpiv(size_P, 1, size_P, 1);
    device_strided_batch_vector<T>   dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_P)
        CHECK_HIP_ERROR(dIpiv.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    int size_W;
    hipsolver_ormqr_unmqr_bufferSize(FORTRAN,
                                     handle,
                                     side,
                                     trans,
                                     m,
                                     n,
                                     k,
                                     dA.data(),
                                     lda,
                                     dIpiv.data(),
                                     dC.data(),
                                     ldc,
                                     &size_W);
    device_strided_batch_vector<T> dWork(size_W, 1, size_W, 1);
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    // check computations
    if(argus.unit_check || argus.norm_check)
        ormqr_unmqr_getError<FORTRAN, T>(handle,
                                         side,
                                         trans,
                                         m,
                                         n,
                                         k,
                                         dA,
                                         lda,
                                         dIpiv,
                                         dC,
                                         ldc,
                                         dWork,
                                         size_W,
                                         dInfo,
                                         hA,
                                         hIpiv,
                                         hC,
                                         hCr,
                                         hInfo,
                                         &max_error);

    // collect performance data
    if(argus.timing)
        ormqr_unmqr_getPerfData<FORTRAN, T>(handle,
                                            side,
                                            trans,
                                            m,
                                            n,
                                            k,
                                            dA,
                                            lda,
                                            dIpiv,
                                            dC,
                                            ldc,
                                            dWork,
                                            size_W,
                                            dInfo,
                                            hA,
                                            hIpiv,
                                            hC,
                                            hInfo,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf);

    // validate results for rocsolver-test
    // using s * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, (left ? m : n));

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("side", "trans", "m", "n", "k", "lda", "ldc");
            rocsolver_bench_output(sideC, transC, m, n, k, lda, ldc);

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
