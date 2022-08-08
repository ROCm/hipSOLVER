/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"

template <testAPI_t API, typename T, typename U>
void sygvj_hegvj_checkBadArgs(const hipsolverHandle_t   handle,
                              const hipsolverEigType_t  itype,
                              const hipsolverEigMode_t  evect,
                              const hipsolverFillMode_t uplo,
                              const int                 n,
                              T                         dA,
                              const int                 lda,
                              const int                 stA,
                              T                         dB,
                              const int                 ldb,
                              const int                 stB,
                              U                         dD,
                              const int                 stD,
                              T                         dWork,
                              const int                 lwork,
                              int*                      dInfo,
                              hipsolverSyevjInfo_t      params,
                              const int                 bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                nullptr,
                                                itype,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                hipsolverEigType_t(-1),
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                hipsolverEigMode_t(-1),
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                evect,
                                                hipsolverFillMode_t(-1),
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_ENUM);

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                evect,
                                                uplo,
                                                n,
                                                (T) nullptr,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                (T) nullptr,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                (U) nullptr,
                                                stD,
                                                dWork,
                                                lwork,
                                                dInfo,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                handle,
                                                itype,
                                                evect,
                                                uplo,
                                                n,
                                                dA,
                                                lda,
                                                stA,
                                                dB,
                                                ldb,
                                                stB,
                                                dD,
                                                stD,
                                                dWork,
                                                lwork,
                                                (int*)nullptr,
                                                params,
                                                bc),
                          HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_sygvj_hegvj_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    hipsolver_local_handle     handle;
    hipsolver_local_syevj_info params;
    int                        n     = 1;
    int                        lda   = 1;
    int                        ldb   = 1;
    int                        stA   = 1;
    int                        stB   = 1;
    int                        stD   = 1;
    int                        bc    = 1;
    hipsolverEigType_t         itype = HIPSOLVER_EIG_TYPE_1;
    hipsolverEigMode_t         evect = HIPSOLVER_EIG_MODE_NOVECTOR;
    hipsolverFillMode_t        uplo  = HIPSOLVER_FILL_MODE_UPPER;

    if(BATCHED)
    {
        // // memory allocations
        // device_batch_vector<T>           dA(1, 1, 1);
        // device_batch_vector<T>           dB(1, 1, 1);
        // device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dB.memcheck());
        // CHECK_HIP_ERROR(dD.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_sygvj_hegvj_bufferSize(API,
        //                                  handle,
        //                                  itype,
        //                                  evect,
        //                                  uplo,
        //                                  n,
        //                                  dA.data(),
        //                                  lda,
        //                                  dB.data(),
        //                                  ldb,
        //                                  dD.data(),
        //                                  &size_W,
        //                                  params);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // sygvj_hegvj_checkBadArgs<API>(handle,
        //                               itype,
        //                               evect,
        //                               uplo,
        //                               n,
        //                               dA.data(),
        //                               lda,
        //                               stA,
        //                               dB.data(),
        //                               ldb,
        //                               stB,
        //                               dD.data(),
        //                               stD,
        //                               dWork.data(),
        //                               size_W,
        //                               dInfo.data(),
        //                               params,
        //                               bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dB(1, 1, 1, 1);
        device_strided_batch_vector<S>   dD(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dB.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_sygvj_hegvj_bufferSize(API,
                                         handle,
                                         itype,
                                         evect,
                                         uplo,
                                         n,
                                         dA.data(),
                                         lda,
                                         dB.data(),
                                         ldb,
                                         dD.data(),
                                         &size_W,
                                         params);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        sygvj_hegvj_checkBadArgs<API>(handle,
                                      itype,
                                      evect,
                                      uplo,
                                      n,
                                      dA.data(),
                                      lda,
                                      stA,
                                      dB.data(),
                                      ldb,
                                      stB,
                                      dD.data(),
                                      stD,
                                      dWork.data(),
                                      size_W,
                                      dInfo.data(),
                                      params,
                                      bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void sygvj_hegvj_initData(const hipsolverHandle_t       handle,
                          const hipsolverEigType_t      itype,
                          const hipsolverEigMode_t      evect,
                          const int                     n,
                          Td&                           dA,
                          const int                     lda,
                          const int                     stA,
                          Td&                           dB,
                          const int                     ldb,
                          const int                     stB,
                          const int                     bc,
                          Th&                           hA,
                          Th&                           hB,
                          host_strided_batch_vector<T>& A,
                          host_strided_batch_vector<T>& B,
                          const bool                    test,
                          const bool                    singular)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);
        rocblas_init<T>(hB, false);

        for(int b = 0; b < bc; ++b)
        {
            for(int i = 0; i < n; i++)
            {
                for(int j = 0; j < n; j++)
                {
                    if(i == j)
                    {
                        hA[b][i + j * lda] = std::real(hA[b][i + j * lda]) + 400;
                        hB[b][i + j * ldb] = std::real(hB[b][i + j * ldb]) + 400;
                    }
                    else
                    {
                        hA[b][i + j * lda] -= 4;
                    }
                }
            }

            // store A and B for testing purposes
            if(test && evect != HIPSOLVER_EIG_MODE_NOVECTOR)
            {
                for(int i = 0; i < n; i++)
                {
                    for(int j = 0; j < n; j++)
                    {
                        if(itype != HIPSOLVER_EIG_TYPE_3)
                        {
                            A[b][i + j * lda] = hA[b][i + j * lda];
                            B[b][i + j * ldb] = hB[b][i + j * ldb];
                        }
                        else
                        {
                            A[b][i + j * lda] = hB[b][i + j * ldb];
                            B[b][i + j * ldb] = hA[b][i + j * lda];
                        }
                    }
                }
            }
        }
    }

    if(GPU)
    {
        // now copy data to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <testAPI_t API,
          typename T,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Uh,
          typename Ih>
void sygvj_hegvj_getError(const hipsolverHandle_t   handle,
                          const hipsolverEigType_t  itype,
                          const hipsolverEigMode_t  evect,
                          const hipsolverFillMode_t uplo,
                          const int                 n,
                          Td&                       dA,
                          const int                 lda,
                          const int                 stA,
                          Td&                       dB,
                          const int                 ldb,
                          const int                 stB,
                          Sd&                       dD,
                          const int                 stD,
                          Td&                       dWork,
                          const int                 lwork,
                          Id&                       dInfo,
                          hipsolverSyevjInfo_t      params,
                          const double              abstol,
                          const int                 max_sweeps,
                          const int                 sort_eig,
                          const int                 bc,
                          Th&                       hA,
                          Th&                       hARes,
                          Th&                       hB,
                          Sh&                       hD,
                          Sh&                       hDRes,
                          Ih&                       hInfo,
                          Ih&                       hInfoRes,
                          Uh&                       hResidualRes,
                          Ih&                       hSweepsRes,
                          double*                   max_err,
                          const bool                singular)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));
    S atol                 = (abstol <= 0) ? get_epsilon<S>() : abstol;

    int lrwork, ltwork;
    if(!COMPLEX)
    {
        lrwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        ltwork = 0;
    }
    else
    {
        lrwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n : 1 + 5 * n + 2 * n * n);
        ltwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : 3 + 5 * n);

    std::vector<T>               work(ltwork);
    std::vector<S>               rwork(lrwork);
    std::vector<int>             iwork(liwork);
    host_strided_batch_vector<T> A(lda * n, 1, lda * n, bc);
    host_strided_batch_vector<T> B(ldb * n, 1, ldb * n, bc);

    // input data initialization
    sygvj_hegvj_initData<true, true, T>(
        handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, true, singular);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_sygvj_hegvj(API,
                                              handle,
                                              itype,
                                              evect,
                                              uplo,
                                              n,
                                              dA.data(),
                                              lda,
                                              stA,
                                              dB.data(),
                                              ldb,
                                              stB,
                                              dD.data(),
                                              stD,
                                              dWork.data(),
                                              lwork,
                                              dInfo.data(),
                                              params,
                                              bc));

    CHECK_HIP_ERROR(hDRes.transfer_from(dD));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));
    if(evect != HIPSOLVER_EIG_MODE_NOVECTOR)
        CHECK_HIP_ERROR(hARes.transfer_from(dA));

    hipsolverXsyevjGetResidual(handle, params, hResidualRes.data());
    hipsolverXsyevjGetSweeps(handle, params, hSweepsRes.data());

    // CPU lapack
    for(int b = 0; b < bc; ++b)
    {
        cblas_sygvd_hegvd(itype,
                          evect,
                          uplo,
                          n,
                          hA[b],
                          lda,
                          hB[b],
                          ldb,
                          hD[b],
                          work.data(),
                          ltwork,
                          rwork.data(),
                          lrwork,
                          iwork.data(),
                          liwork,
                          hInfo[b]);
    }

    // check info for non-convergence and/or positive-definiteness
    *max_err = 0;
    for(int b = 0; b < bc; ++b)
        if(hInfo[b][0] != hInfoRes[b][0])
            *max_err += 1;

    // Also check validity of residual
    for(rocblas_int b = 0; b < bc; ++b)
        if(hResidualRes[b][0] < 0 || hResidualRes[b][0] > atol)
            *max_err += 1;

    // Also check validity of sweeps
    for(rocblas_int b = 0; b < bc; ++b)
        if(hSweepsRes[b][0] < 0 || hSweepsRes[b][0] > max_sweeps)
            *max_err += 1;

    double err;

    if(evect == HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        // only eigenvalues needed; can compare with LAPACK

        // error is ||hD - hDRes|| / ||hD||
        // using frobenius norm
        for(int b = 0; b < bc; ++b)
        {
            if(hInfoRes[b][0] == 0 && sort_eig)
            {
                err      = norm_error('F', 1, n, 1, hD[b], hDRes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
    else
    {
        // both eigenvalues and eigenvectors needed; need to implicitly test
        // eigenvectors due to non-uniqueness of eigenvectors under scaling

        for(int b = 0; b < bc; ++b)
        {
            if(hInfoRes[b][0] == 0)
            {
                T alpha = 1;
                T beta  = 0;

                // hARes contains eigenvectors x
                // compute B*x (or A*x) and store in hB
                cblas_symm_hemm<T>(HIPSOLVER_SIDE_LEFT,
                                   uplo,
                                   n,
                                   n,
                                   alpha,
                                   B[b],
                                   ldb,
                                   hARes[b],
                                   lda,
                                   beta,
                                   hB[b],
                                   ldb);

                if(itype == HIPSOLVER_EIG_TYPE_1)
                {
                    // problem is A*x = (lambda)*B*x

                    // compute (1/lambda)*A*x and store in hA
                    for(int j = 0; j < n; j++)
                    {
                        alpha = T(1) / hDRes[b][j];
                        cblas_symv_hemv(uplo,
                                        n,
                                        alpha,
                                        A[b],
                                        lda,
                                        hARes[b] + j * lda,
                                        1,
                                        beta,
                                        hA[b] + j * lda,
                                        1);
                    }

                    // move B*x into hARes
                    for(int i = 0; i < n; i++)
                        for(int j = 0; j < n; j++)
                            hARes[b][i + j * lda] = hB[b][i + j * ldb];
                }
                else
                {
                    // problem is A*B*x = (lambda)*x or B*A*x = (lambda)*x

                    // compute (1/lambda)*A*B*x or (1/lambda)*B*A*x and store in hA
                    for(int j = 0; j < n; j++)
                    {
                        alpha = T(1) / hDRes[b][j];
                        cblas_symv_hemv(uplo,
                                        n,
                                        alpha,
                                        A[b],
                                        lda,
                                        hB[b] + j * ldb,
                                        1,
                                        beta,
                                        hA[b] + j * lda,
                                        1);
                    }
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err      = norm_error('F', n, n, lda, hA[b], hARes[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <testAPI_t API,
          typename T,
          typename Sd,
          typename Td,
          typename Id,
          typename Sh,
          typename Th,
          typename Ih>
void sygvj_hegvj_getPerfData(const hipsolverHandle_t   handle,
                             const hipsolverEigType_t  itype,
                             const hipsolverEigMode_t  evect,
                             const hipsolverFillMode_t uplo,
                             const int                 n,
                             Td&                       dA,
                             const int                 lda,
                             const int                 stA,
                             Td&                       dB,
                             const int                 ldb,
                             const int                 stB,
                             Sd&                       dD,
                             const int                 stD,
                             Td&                       dWork,
                             const int                 lwork,
                             Id&                       dInfo,
                             hipsolverSyevjInfo_t      params,
                             const int                 bc,
                             Th&                       hA,
                             Th&                       hB,
                             Sh&                       hD,
                             Ih&                       hInfo,
                             double*                   gpu_time_used,
                             double*                   cpu_time_used,
                             const int                 hot_calls,
                             const bool                perf,
                             const bool                singular)
{
    constexpr bool COMPLEX = is_complex<T>;
    using S                = decltype(std::real(T{}));

    int lrwork, ltwork;
    if(!COMPLEX)
    {
        lrwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 2 * n + 1 : 1 + 6 * n + 2 * n * n);
        ltwork = 0;
    }
    else
    {
        lrwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n : 1 + 5 * n + 2 * n * n);
        ltwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? n + 1 : 2 * n + n * n);
    }
    int liwork = (evect == HIPSOLVER_EIG_MODE_NOVECTOR ? 1 : 3 + 5 * n);

    std::vector<T>               work(ltwork);
    std::vector<S>               rwork(lrwork);
    std::vector<int>             iwork(liwork);
    host_strided_batch_vector<T> A(1, 1, 1, 1);
    host_strided_batch_vector<T> B(1, 1, 1, 1);

    if(!perf)
    {
        sygvj_hegvj_initData<true, false, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false, singular);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
        {
            cblas_sygvd_hegvd<T>(itype,
                                 evect,
                                 uplo,
                                 n,
                                 hA[b],
                                 lda,
                                 hB[b],
                                 ldb,
                                 hD[b],
                                 work.data(),
                                 ltwork,
                                 rwork.data(),
                                 lrwork,
                                 iwork.data(),
                                 liwork,
                                 hInfo[b]);
        }
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    sygvj_hegvj_initData<true, false, T>(
        handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        sygvj_hegvj_initData<false, true, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false, singular);

        CHECK_ROCBLAS_ERROR(hipsolver_sygvj_hegvj(API,
                                                  handle,
                                                  itype,
                                                  evect,
                                                  uplo,
                                                  n,
                                                  dA.data(),
                                                  lda,
                                                  stA,
                                                  dB.data(),
                                                  ldb,
                                                  stB,
                                                  dD.data(),
                                                  stD,
                                                  dWork.data(),
                                                  lwork,
                                                  dInfo.data(),
                                                  params,
                                                  bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        sygvj_hegvj_initData<false, true, T>(
            handle, itype, evect, n, dA, lda, stA, dB, ldb, stB, bc, hA, hB, A, B, false, singular);

        start = get_time_us_sync(stream);
        hipsolver_sygvj_hegvj(API,
                              handle,
                              itype,
                              evect,
                              uplo,
                              n,
                              dA.data(),
                              lda,
                              stA,
                              dB.data(),
                              ldb,
                              stB,
                              dD.data(),
                              stD,
                              dWork.data(),
                              lwork,
                              dInfo.data(),
                              params,
                              bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_sygvj_hegvj(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolver_local_handle     handle;
    hipsolver_local_syevj_info params;
    char                       itypeC = argus.get<char>("itype");
    char                       evectC = argus.get<char>("jobz");
    char                       uploC  = argus.get<char>("uplo");
    int                        n      = argus.get<int>("n");
    int                        lda    = argus.get<int>("lda", n);
    int                        ldb    = argus.get<int>("ldb", n);
    int                        stA    = argus.get<int>("strideA", lda * n);
    int                        stB    = argus.get<int>("strideB", ldb * n);
    int                        stD    = argus.get<int>("strideD", n);

    double      abstol     = argus.get<double>("tolerance", 2 * get_epsilon<T>());
    rocblas_int max_sweeps = argus.get<int>("max_sweeps", 100);
    rocblas_int sort_eig   = argus.get<int>("sort_eig", 1);
    hipsolverXsyevjSetTolerance(params, abstol);
    hipsolverXsyevjSetMaxSweeps(params, max_sweeps);
    hipsolverXsyevjSetSortEig(params, sort_eig);

    hipsolverEigType_t  itype     = char2hipsolver_eform(itypeC);
    hipsolverEigMode_t  evect     = char2hipsolver_evect(evectC);
    hipsolverFillMode_t uplo      = char2hipsolver_fill(uploC);
    int                 bc        = argus.batch_count;
    int                 hot_calls = argus.iters;

    int stARes = (argus.unit_check || argus.norm_check) ? stA : 0;
    int stDRes = (argus.unit_check || argus.norm_check) ? stD : 0;

    // determine sizes
    size_t size_A    = size_t(lda) * n;
    size_t size_B    = size_t(ldb) * n;
    size_t size_D    = size_t(n);
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_DRes = (argus.unit_check || argus.norm_check) ? size_D : 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || ldb < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
        {
            // EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
            //                                             handle,
            //                                             itype,
            //                                             evect,
            //                                             uplo,
            //                                             n,
            //                                             (T* const*)nullptr,
            //                                             lda,
            //                                             stA,
            //                                             (T* const*)nullptr,
            //                                             ldb,
            //                                             stB,
            //                                             (S*)nullptr,
            //                                             stD,
            //                                             (T*)nullptr,
            //                                             0,
            //                                             (int*)nullptr,
            //                                             params,
            //                                             bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_sygvj_hegvj(API,
                                                        handle,
                                                        itype,
                                                        evect,
                                                        uplo,
                                                        n,
                                                        (T*)nullptr,
                                                        lda,
                                                        stA,
                                                        (T*)nullptr,
                                                        ldb,
                                                        stB,
                                                        (S*)nullptr,
                                                        stD,
                                                        (T*)nullptr,
                                                        0,
                                                        (int*)nullptr,
                                                        params,
                                                        bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_sygvj_hegvj_bufferSize(API,
                                     handle,
                                     itype,
                                     evect,
                                     uplo,
                                     n,
                                     (T*)nullptr,
                                     lda,
                                     (T*)nullptr,
                                     ldb,
                                     (S*)nullptr,
                                     &size_W,
                                     params);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    // memory allocations
    // host
    host_strided_batch_vector<double> hResidualRes(1, 1, 1, bc);
    host_strided_batch_vector<int>    hSweepsRes(1, 1, 1, bc);
    host_strided_batch_vector<S>      hD(size_D, 1, stD, bc);
    host_strided_batch_vector<S>      hDRes(size_DRes, 1, stDRes, bc);
    host_strided_batch_vector<int>    hInfo(1, 1, 1, bc);
    host_strided_batch_vector<int>    hInfoRes(1, 1, 1, bc);
    // device
    device_strided_batch_vector<S>   dD(size_D, 1, stD, bc);
    device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
    device_strided_batch_vector<T>   dWork(size_W, 1, size_W, bc);
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());
    if(size_W)
        CHECK_HIP_ERROR(dWork.memcheck());

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hARes(size_ARes, 1, bc);
        // host_batch_vector<T>             hB(size_B, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_batch_vector<T>           dB(size_B, 1, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_B)
        //     CHECK_HIP_ERROR(dB.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     sygvj_hegvj_getError<API, T>(handle,
        //                                  itype,
        //                                  evect,
        //                                  uplo,
        //                                  n,
        //                                  dA,
        //                                  lda,
        //                                  stA,
        //                                  dB,
        //                                  ldb,
        //                                  stB,
        //                                  dD,
        //                                  stD,
        //                                  dWork,
        //                                  size_W,
        //                                  dInfo,
        //                                  params,
        //                                  abstol,
        //                                  max_sweeps,
        //                                  sort_eig,
        //                                  bc,
        //                                  hA,
        //                                  hARes,
        //                                  hB,
        //                                  hD,
        //                                  hDRes,
        //                                  hInfo,
        //                                  hInfoRes,
        //                                  hResidualRes,
        //                                  hSweepsRes,
        //                                  &max_error,
        //                                  argus.singular);

        // // collect performance data
        // if(argus.timing)
        //     sygvj_hegvj_getPerfData<API, T>(handle,
        //                                     itype,
        //                                     evect,
        //                                     uplo,
        //                                     n,
        //                                     dA,
        //                                     lda,
        //                                     stA,
        //                                     dB,
        //                                     ldb,
        //                                     stB,
        //                                     dD,
        //                                     stD,
        //                                     dWork,
        //                                     size_W,
        //                                     dInfo,
        //                                     params,
        //                                     bc,
        //                                     hA,
        //                                     hB,
        //                                     hD,
        //                                     hInfo,
        //                                     &gpu_time_used,
        //                                     &cpu_time_used,
        //                                     hot_calls,
        //                                     argus.perf,
        //                                     argus.singular);
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T>   hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T>   hARes(size_ARes, 1, stARes, bc);
        host_strided_batch_vector<T>   hB(size_B, 1, stB, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T> dB(size_B, 1, stB, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_B)
            CHECK_HIP_ERROR(dB.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            sygvj_hegvj_getError<API, T>(handle,
                                         itype,
                                         evect,
                                         uplo,
                                         n,
                                         dA,
                                         lda,
                                         stA,
                                         dB,
                                         ldb,
                                         stB,
                                         dD,
                                         stD,
                                         dWork,
                                         size_W,
                                         dInfo,
                                         params,
                                         abstol,
                                         max_sweeps,
                                         sort_eig,
                                         bc,
                                         hA,
                                         hARes,
                                         hB,
                                         hD,
                                         hDRes,
                                         hInfo,
                                         hInfoRes,
                                         hResidualRes,
                                         hSweepsRes,
                                         &max_error,
                                         argus.singular);

        // collect performance data
        if(argus.timing)
            sygvj_hegvj_getPerfData<API, T>(handle,
                                            itype,
                                            evect,
                                            uplo,
                                            n,
                                            dA,
                                            lda,
                                            stA,
                                            dB,
                                            ldb,
                                            stB,
                                            dD,
                                            stD,
                                            dWork,
                                            size_W,
                                            dInfo,
                                            params,
                                            bc,
                                            hA,
                                            hB,
                                            hD,
                                            hInfo,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf,
                                            argus.singular);
    }

    // validate results for rocsolver-test
    // using 2 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 2 * n);

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
                rocsolver_bench_output("itype",
                                       "evect",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "ldb",
                                       "strideD",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(
                    itypeC, evectC, uploC, n, lda, ldb, stD, abstol, max_sweeps, sort_eig, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("itype",
                                       "evect",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "ldb",
                                       "strideA",
                                       "strideB",
                                       "strideD",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig",
                                       "batch_c");
                rocsolver_bench_output(itypeC,
                                       evectC,
                                       uploC,
                                       n,
                                       lda,
                                       ldb,
                                       stA,
                                       stB,
                                       stD,
                                       abstol,
                                       max_sweeps,
                                       sort_eig,
                                       bc);
            }
            else
            {
                rocsolver_bench_output("itype",
                                       "evect",
                                       "uplo",
                                       "n",
                                       "lda",
                                       "ldb",
                                       "tolerance",
                                       "max_sweeps",
                                       "sort_eig");
                rocsolver_bench_output(
                    itypeC, evectC, uploC, n, lda, ldb, abstol, max_sweeps, sort_eig);
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
