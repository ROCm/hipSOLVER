/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

template <testAPI_t API, typename T, typename U, typename V>
void geqrf_checkBadArgs(const hipsolverHandle_t handle,
                        const int               m,
                        const int               n,
                        T                       dA,
                        const int               lda,
                        const int               stA,
                        U                       dIpiv,
                        const int               stP,
                        U                       dWork,
                        const int               lwork,
                        V                       dInfo,
                        const int               bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_geqrf(API, nullptr, m, n, dA, lda, stA, dIpiv, stP, dWork, lwork, dInfo, bc),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_geqrf(
            API, handle, m, n, (T) nullptr, lda, stA, dIpiv, stP, dWork, lwork, dInfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_geqrf(API, handle, m, n, dA, lda, stA, (U) nullptr, stP, dWork, lwork, dInfo, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_geqrf(API, handle, m, n, dA, lda, stA, dIpiv, stP, dWork, lwork, (V) nullptr, bc),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_geqrf_bad_arg()
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
        // device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        // CHECK_HIP_ERROR(dA.memcheck());
        // CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());

        // int size_W;
        // hipsolver_geqrf_bufferSize(API, handle, m, n, dA.data(), lda, &size_W);
        // device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check bad arguments
        // geqrf_checkBadArgs<API>(handle,
        //                             m,
        //                             n,
        //                             dA.data(),
        //                             lda,
        //                             stA,
        //                             dIpiv.data(),
        //                             stP,
        //                             dWork.data(),
        //                             size_W,
        //                             dInfo.data(),
        //                             bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T>   dA(1, 1, 1, 1);
        device_strided_batch_vector<T>   dIpiv(1, 1, 1, 1);
        device_strided_batch_vector<int> dInfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());

        int size_W;
        hipsolver_geqrf_bufferSize(API, handle, m, n, dA.data(), lda, &size_W);
        device_strided_batch_vector<T> dWork(size_W, 1, size_W, bc);
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check bad arguments
        geqrf_checkBadArgs<API>(handle,
                                m,
                                n,
                                dA.data(),
                                lda,
                                stA,
                                dIpiv.data(),
                                stP,
                                dWork.data(),
                                size_W,
                                dInfo.data(),
                                bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void geqrf_initData(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Ud&                     dIpiv,
                    const int               stP,
                    const int               bc,
                    Th&                     hA,
                    Uh&                     hIpiv)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
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
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh,
          typename Vh>
void geqrf_getError(const hipsolverHandle_t handle,
                    const int               m,
                    const int               n,
                    Td&                     dA,
                    const int               lda,
                    const int               stA,
                    Ud&                     dIpiv,
                    const int               stP,
                    Ud&                     dWork,
                    const int               lwork,
                    Vd&                     dInfo,
                    const int               bc,
                    Th&                     hA,
                    Th&                     hARes,
                    Uh&                     hIpiv,
                    Vh&                     hInfo,
                    Vh&                     hInfoRes,
                    double*                 max_err)
{
    std::vector<T> hW(n);

    // input data initialization
    geqrf_initData<true, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_geqrf(API,
                                        handle,
                                        m,
                                        n,
                                        dA.data(),
                                        lda,
                                        stA,
                                        dIpiv.data(),
                                        stP,
                                        dWork.data(),
                                        lwork,
                                        dInfo.data(),
                                        bc));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // CPU lapack
    for(int b = 0; b < bc; ++b)
        cpu_geqrf(m, n, hA[b], lda, hIpiv[b], hW.data(), n, hInfo[b]);

    // error is ||hA - hARes|| / ||hA|| (ideally ||QR - Qres Rres|| / ||QR||)
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

    // check info
    err = 0;
    for(int b = 0; b < bc; ++b)
    {
        EXPECT_EQ(hInfo[b][0], hInfoRes[b][0]) << "where b = " << b;
        if(hInfo[b][0] != hInfoRes[b][0])
            err++;
    }
    *max_err += err;
}

template <testAPI_t API,
          typename T,
          typename Td,
          typename Ud,
          typename Vd,
          typename Th,
          typename Uh,
          typename Vh>
void geqrf_getPerfData(const hipsolverHandle_t handle,
                       const int               m,
                       const int               n,
                       Td&                     dA,
                       const int               lda,
                       const int               stA,
                       Ud&                     dIpiv,
                       const int               stP,
                       Ud&                     dWork,
                       const int               lwork,
                       Vd&                     dInfo,
                       const int               bc,
                       Th&                     hA,
                       Uh&                     hIpiv,
                       Vh&                     hInfo,
                       double*                 gpu_time_used,
                       double*                 cpu_time_used,
                       const int               hot_calls,
                       const bool              perf)
{
    std::vector<T> hW(n);

    if(!perf)
    {
        geqrf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(int b = 0; b < bc; ++b)
            cpu_geqrf(m, n, hA[b], lda, hIpiv[b], hW.data(), n, hInfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    geqrf_initData<true, false, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geqrf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        CHECK_ROCBLAS_ERROR(hipsolver_geqrf(API,
                                            handle,
                                            m,
                                            n,
                                            dA.data(),
                                            lda,
                                            stA,
                                            dIpiv.data(),
                                            stP,
                                            dWork.data(),
                                            lwork,
                                            dInfo.data(),
                                            bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(hipsolverGetStream(handle, &stream));
    double start;

    for(int iter = 0; iter < hot_calls; iter++)
    {
        geqrf_initData<false, true, T>(handle, m, n, dA, lda, stA, dIpiv, stP, bc, hA, hIpiv);

        start = get_time_us_sync(stream);
        hipsolver_geqrf(API,
                        handle,
                        m,
                        n,
                        dA.data(),
                        lda,
                        stA,
                        dIpiv.data(),
                        stP,
                        dWork.data(),
                        lwork,
                        dInfo.data(),
                        bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <testAPI_t API, bool BATCHED, bool STRIDED, typename T>
void testing_geqrf(Arguments& argus)
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
            // EXPECT_ROCBLAS_STATUS(hipsolver_geqrf(API,
            //                                       handle,
            //                                       m,
            //                                       n,
            //                                       (T* const*)nullptr,
            //                                       lda,
            //                                       stA,
            //                                       (T*)nullptr,
            //                                       stP,
            //                                       (T*)nullptr,
            //                                       0,
            //                                       (int*)nullptr,
            //                                       bc),
            //                       HIPSOLVER_STATUS_INVALID_VALUE);
        }
        else
        {
            EXPECT_ROCBLAS_STATUS(hipsolver_geqrf(API,
                                                  handle,
                                                  m,
                                                  n,
                                                  (T*)nullptr,
                                                  lda,
                                                  stA,
                                                  (T*)nullptr,
                                                  stP,
                                                  (T*)nullptr,
                                                  0,
                                                  (int*)nullptr,
                                                  bc),
                                  HIPSOLVER_STATUS_INVALID_VALUE);
        }

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    int size_W;
    hipsolver_geqrf_bufferSize(API, handle, m, n, (T*)nullptr, lda, &size_W);

    if(argus.mem_query)
    {
        rocsolver_bench_inform(inform_mem_query, size_W);
        return;
    }

    if(BATCHED)
    {
        // // memory allocations
        // host_batch_vector<T>             hA(size_A, 1, bc);
        // host_batch_vector<T>             hARes(size_ARes, 1, bc);
        // host_strided_batch_vector<T>     hIpiv(size_P, 1, stP, bc);
        // host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        // host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        // device_batch_vector<T>           dA(size_A, 1, bc);
        // device_strided_batch_vector<T>   dIpiv(size_P, 1, stP, bc);
        // device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        // device_strided_batch_vector<T>   dWork(size_W, 1, size_W, bc);
        // if(size_A)
        //     CHECK_HIP_ERROR(dA.memcheck());
        // if(size_P)
        //     CHECK_HIP_ERROR(dIpiv.memcheck());
        // CHECK_HIP_ERROR(dInfo.memcheck());
        // if(size_W)
        //     CHECK_HIP_ERROR(dWork.memcheck());

        // // check computations
        // if(argus.unit_check || argus.norm_check)
        //     geqrf_getError<API, T>(handle,
        //                                m,
        //                                n,
        //                                dA,
        //                                lda,
        //                                stA,
        //                                dIpiv,
        //                                stP,
        //                                dWork,
        //                                size_W,
        //                                dInfo,
        //                                bc,
        //                                hA,
        //                                hARes,
        //                                hIpiv,
        //                                hInfo,
        //                                hInfoRes,
        //                                &max_error);

        // // collect performance data
        // if(argus.timing)
        //     geqrf_getPerfData<API, T>(handle,
        //                                   m,
        //                                   n,
        //                                   dA,
        //                                   lda,
        //                                   stA,
        //                                   dIpiv,
        //                                   stP,
        //                                   dWork,
        //                                   size_W,
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
        host_strided_batch_vector<T>     hIpiv(size_P, 1, stP, bc);
        host_strided_batch_vector<int>   hInfo(1, 1, 1, bc);
        host_strided_batch_vector<int>   hInfoRes(1, 1, 1, bc);
        device_strided_batch_vector<T>   dA(size_A, 1, stA, bc);
        device_strided_batch_vector<T>   dIpiv(size_P, 1, stP, bc);
        device_strided_batch_vector<int> dInfo(1, 1, 1, bc);
        device_strided_batch_vector<T>   dWork(size_W, 1, size_W, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());
        if(size_P)
            CHECK_HIP_ERROR(dIpiv.memcheck());
        CHECK_HIP_ERROR(dInfo.memcheck());
        if(size_W)
            CHECK_HIP_ERROR(dWork.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            geqrf_getError<API, T>(handle,
                                   m,
                                   n,
                                   dA,
                                   lda,
                                   stA,
                                   dIpiv,
                                   stP,
                                   dWork,
                                   size_W,
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
            geqrf_getPerfData<API, T>(handle,
                                      m,
                                      n,
                                      dA,
                                      lda,
                                      stA,
                                      dIpiv,
                                      stP,
                                      dWork,
                                      size_W,
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
    // using m * machine_precision as tolerance
    // (for possibly singular of ill-conditioned matrices we could use m*min(m,n))
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
