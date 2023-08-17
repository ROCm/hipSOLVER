/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "hipsolverSp.hpp"

template <bool HOST, typename T>
void csrlsvchol_checkBadArgs(hipsolverSpHandle_t       handle,
                             const int                 n,
                             const int                 nnzA,
                             const hipsparseMatDescr_t descrA,
                             int*                      ptrA,
                             int*                      indA,
                             T                         valA,
                             T                         B,
                             T                         X,
                             int*                      singularity)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, nullptr, n, nnzA, descrA, valA, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_NOT_INITIALIZED);

    // values
    // N/A

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
    // pointers
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, nullptr, valA, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, descrA, (T) nullptr, ptrA, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, descrA, valA, (int*)nullptr, indA, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, descrA, valA, ptrA, (int*)nullptr, B, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, descrA, valA, ptrA, indA, (T) nullptr, 0, 0, X, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
    EXPECT_ROCBLAS_STATUS(
        hipsolver_csrlsvchol(
            HOST, handle, n, nnzA, descrA, valA, ptrA, indA, B, 0, 0, (T) nullptr, singularity),
        HIPSOLVER_STATUS_INVALID_VALUE);
#endif
}

template <bool HOST, typename T>
void testing_csrlsvchol_bad_arg()
{
    // safe arguments
    hipsolverSp_local_handle handle;
    int                      n    = 1;
    int                      nnzA = 1;

    hipsparse_local_mat_descr descrA;
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);

    if(HOST)
    {
        // memory allocations
        host_strided_batch_vector<int> singularity(1, 1, 1, 1);
        host_strided_batch_vector<int> ptrA(1, 1, 1, 1);
        host_strided_batch_vector<int> indA(1, 1, 1, 1);
        host_strided_batch_vector<T>   valA(1, 1, 1, 1);
        host_strided_batch_vector<T>   B(1, 1, 1, 1);
        host_strided_batch_vector<T>   X(1, 1, 1, 1);

        // check bad arguments
        csrlsvchol_checkBadArgs<HOST>(handle,
                                      n,
                                      nnzA,
                                      descrA,
                                      ptrA.data(),
                                      indA.data(),
                                      valA.data(),
                                      B.data(),
                                      X.data(),
                                      singularity.data());
    }
    else
    {
        // memory allocations
        host_strided_batch_vector<int>   singularity(1, 1, 1, 1);
        device_strided_batch_vector<int> ptrA(1, 1, 1, 1);
        device_strided_batch_vector<int> indA(1, 1, 1, 1);
        device_strided_batch_vector<T>   valA(1, 1, 1, 1);
        device_strided_batch_vector<T>   B(1, 1, 1, 1);
        device_strided_batch_vector<T>   X(1, 1, 1, 1);
        CHECK_HIP_ERROR(ptrA.memcheck());
        CHECK_HIP_ERROR(indA.memcheck());
        CHECK_HIP_ERROR(valA.memcheck());
        CHECK_HIP_ERROR(B.memcheck());
        CHECK_HIP_ERROR(X.memcheck());

        // check bad arguments
        csrlsvchol_checkBadArgs<HOST>(handle,
                                      n,
                                      nnzA,
                                      descrA,
                                      ptrA.data(),
                                      indA.data(),
                                      valA.data(),
                                      B.data(),
                                      X.data(),
                                      singularity.data());
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvchol_initData(hipsolverSpHandle_t handle,
                         const int           n,
                         const int           nnzA,
                         Ud&                 dptrA,
                         Ud&                 dindA,
                         Td&                 dvalA,
                         const int           nnzT,
                         Ud&                 dptrT,
                         Ud&                 dindT,
                         Td&                 dvalT,
                         Td&                 dB,
                         Uh&                 hptrA,
                         Uh&                 hindA,
                         Th&                 hvalA,
                         Uh&                 hptrT,
                         Uh&                 hindT,
                         Th&                 hvalT,
                         Th&                 hB,
                         Th&                 hX,
                         const fs::path      testcase,
                         bool                test = true)
{
    if(CPU)
    {
        fs::path file;

        // read-in A
        file = testcase / "ptrA";
        read_matrix(file.string(), 1, n + 1, hptrA.data(), 1);
        file = testcase / "indA";
        read_matrix(file.string(), 1, nnzA, hindA.data(), 1);
        file = testcase / "valA";
        read_matrix(file.string(), 1, nnzA, hvalA.data(), 1);

        // read-in T
        file = testcase / "ptrT";
        read_matrix(file.string(), 1, n + 1, hptrT.data(), 1);
        file = testcase / "indT";
        read_matrix(file.string(), 1, nnzT, hindT.data(), 1);
        file = testcase / "valT";
        read_matrix(file.string(), 1, nnzT, hvalT.data(), 1);

        // read-in B
        file = testcase / "B_1";
        read_matrix(file.string(), n, 1, hB.data(), n);

        // get results (matrix X) if validation is required
        if(test)
        {
            // read-in X
            file = testcase / "X_1";
            read_matrix(file.string(), n, 1, hX.data(), n);
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dptrA.transfer_from(hptrA));
        CHECK_HIP_ERROR(dindA.transfer_from(hindA));
        CHECK_HIP_ERROR(dvalA.transfer_from(hvalA));
        CHECK_HIP_ERROR(dptrT.transfer_from(hptrT));
        CHECK_HIP_ERROR(dindT.transfer_from(hindT));
        CHECK_HIP_ERROR(dvalT.transfer_from(hvalT));
        CHECK_HIP_ERROR(dB.transfer_from(hB));
    }
}

template <bool HOST, typename T, typename S, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvchol_getError(hipsolverSpHandle_t       handle,
                         const int                 n,
                         const int                 nnzA,
                         const hipsparseMatDescr_t descrA,
                         Ud&                       dptrA,
                         Ud&                       dindA,
                         Td&                       dvalA,
                         const int                 nnzT,
                         Ud&                       dptrT,
                         Ud&                       dindT,
                         Td&                       dvalT,
                         Td&                       dB,
                         const S                   tolerance,
                         const int                 reorder,
                         Td&                       dX,
                         Uh&                       hptrA,
                         Uh&                       hindA,
                         Th&                       hvalA,
                         Uh&                       hptrT,
                         Uh&                       hptrTRes,
                         Uh&                       hindT,
                         Uh&                       hindTRes,
                         Th&                       hvalT,
                         Th&                       hvalTRes,
                         Th&                       hB,
                         Th&                       hX,
                         Th&                       hXRes,
                         Uh&                       hSingularity,
                         double*                   max_err,
                         const fs::path            testcase)
{
    // input data initialization
    csrlsvchol_initData<true, true, T>(handle,
                                       n,
                                       nnzA,
                                       dptrA,
                                       dindA,
                                       dvalA,
                                       nnzT,
                                       dptrT,
                                       dindT,
                                       dvalT,
                                       dB,
                                       hptrA,
                                       hindA,
                                       hvalA,
                                       hptrT,
                                       hindT,
                                       hvalT,
                                       hB,
                                       hX,
                                       testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(hipsolver_csrlsvchol(HOST,
                                             handle,
                                             n,
                                             nnzA,
                                             descrA,
                                             dvalA.data(),
                                             dptrA.data(),
                                             dindA.data(),
                                             dB.data(),
                                             tolerance,
                                             reorder,
                                             dX.data(),
                                             hSingularity.data()));

    CHECK_HIP_ERROR(hptrTRes.transfer_from(dptrT));
    CHECK_HIP_ERROR(hindTRes.transfer_from(dindT));
    CHECK_HIP_ERROR(hvalTRes.transfer_from(dvalT));
    CHECK_HIP_ERROR(hXRes.transfer_from(dX));

    // compare computed results with original result
    double err;
    *max_err = 0;

    err      = norm_error('I', 1, n + 1, 1, hptrT[0], hptrTRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    err      = norm_error('I', 1, nnzT, 1, hindT[0], hindTRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    err      = norm_error('I', 1, nnzT, 1, hvalT[0], hvalTRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    err      = norm_error('I', n, 1, n, hX[0], hXRes[0]);
    *max_err = err > *max_err ? err : *max_err;

    // TODO: Add non-positive definite test matrices
    // also check info for singularities
    err = 0;
    EXPECT_EQ(hSingularity[0][0], -1);
    if(hSingularity[0][0] != -1)
        err++;
    *max_err += err;
}

template <bool HOST, typename T, typename S, typename Td, typename Ud, typename Th, typename Uh>
void csrlsvchol_getPerfData(hipsolverSpHandle_t       handle,
                            const int                 n,
                            const int                 nnzA,
                            const hipsparseMatDescr_t descrA,
                            Ud&                       dptrA,
                            Ud&                       dindA,
                            Td&                       dvalA,
                            const int                 nnzT,
                            Ud&                       dptrT,
                            Ud&                       dindT,
                            Td&                       dvalT,
                            Td&                       dB,
                            const S                   tolerance,
                            const int                 reorder,
                            Td&                       dX,
                            Uh&                       hptrA,
                            Uh&                       hindA,
                            Th&                       hvalA,
                            Uh&                       hptrT,
                            Uh&                       hindT,
                            Th&                       hvalT,
                            Th&                       hB,
                            Th&                       hX,
                            Uh&                       hSingularity,
                            double*                   gpu_time_used,
                            double*                   cpu_time_used,
                            const int                 hot_calls,
                            const bool                perf,
                            const fs::path            testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution
    *gpu_time_used = nan(""); // no timing on gpu-lapack execution
}

template <bool HOST, typename T>
void testing_csrlsvchol(Arguments& argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    hipsolverSp_local_handle handle;
    int                      n         = argus.get<int>("n");
    int                      nnzA      = argus.get<int>("nnzA");
    double                   tolerance = argus.get<double>("tolerance", 0);
    int                      reorder   = argus.get<int>("reorder", 0);
    int                      hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzA < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(hipsolver_csrlsvchol(HOST,
                                                   handle,
                                                   n,
                                                   nnzA,
                                                   (hipsparseMatDescr_t) nullptr,
                                                   (T*)nullptr,
                                                   (int*)nullptr,
                                                   (int*)nullptr,
                                                   (T*)nullptr,
                                                   tolerance,
                                                   reorder,
                                                   (T*)nullptr,
                                                   (int*)nullptr),
                              HIPSOLVER_STATUS_INVALID_VALUE);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine existing test case
    if(n > 0)
    {
        if(n <= 35)
            n = 20;
        else if(n <= 75)
            n = 50;
        else if(n <= 175)
            n = 100;
        else
            n = 250;
    }

    if(n <= 50) // small case
    {
        if(nnzA <= 80)
            nnzA = 60;
        else if(nnzA <= 120)
            nnzA = 100;
        else
            nnzA = 140;
    }
    else // large case
    {
        if(nnzA <= 400)
            nnzA = 300;
        else if(nnzA <= 600)
            nnzA = 500;
        else
            nnzA = 700;
    }

    // read/set corresponding nnzT
    int      nnzT;
    fs::path testcase;
    if(n > 0)
    {
        fs::path    file;
        std::string folder
            = std::string("posmat_") + std::to_string(n) + "_" + std::to_string(nnzA);
        testcase = get_sparse_data_dir() / folder;

        file = testcase / "ptrT";
        read_last(file.string(), &nnzT);
    }

    // determine sizes
    size_t size_ptrA = size_t(n) + 1;
    size_t size_indA = size_t(nnzA);
    size_t size_valA = size_t(nnzA);
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);
    size_t size_BX   = size_t(n);

    size_t size_ptrTRes = 0;
    size_t size_indTRes = 0;
    size_t size_valTRes = 0;
    size_t size_BXres   = 0;
    if(argus.unit_check || argus.norm_check)
    {
        size_ptrTRes = size_ptrT;
        size_indTRes = size_indT;
        size_valTRes = size_valT;
        size_BXres   = size_BX;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations (all cases)
    hipsparse_local_mat_descr descrA;
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);

    host_strided_batch_vector<int> hptrA(size_ptrA, 1, size_ptrA, 1);
    host_strided_batch_vector<int> hindA(size_indA, 1, size_indA, 1);
    host_strided_batch_vector<T>   hvalA(size_valA, 1, size_valA, 1);
    host_strided_batch_vector<int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<int> hptrTRes(size_ptrTRes, 1, size_ptrTRes, 1);
    host_strided_batch_vector<int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<int> hindTRes(size_indTRes, 1, size_indTRes, 1);
    host_strided_batch_vector<T>   hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<T>   hvalTRes(size_valTRes, 1, size_valTRes, 1);
    host_strided_batch_vector<T>   hB(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hX(size_BX, 1, size_BX, 1);
    host_strided_batch_vector<T>   hXRes(size_BXres, 1, size_BXres, 1);
    host_strided_batch_vector<int> hSingularity(1, 1, 1, 1);

    if(HOST)
    {
        // memory allocations
        host_strided_batch_vector<int> dptrA(size_ptrA, 1, size_ptrA, 1);
        host_strided_batch_vector<int> dindA(size_indA, 1, size_indA, 1);
        host_strided_batch_vector<T>   dvalA(size_valA, 1, size_valA, 1);
        host_strided_batch_vector<int> dptrT(size_ptrT, 1, size_ptrT, 1);
        host_strided_batch_vector<int> dindT(size_indT, 1, size_indT, 1);
        host_strided_batch_vector<T>   dvalT(size_valT, 1, size_valT, 1);
        host_strided_batch_vector<T>   dB(size_BX, 1, size_BX, 1);
        host_strided_batch_vector<T>   dX(size_BX, 1, size_BX, 1);

        // check computations
        if(argus.unit_check || argus.norm_check)
            csrlsvchol_getError<HOST, T>(handle,
                                         n,
                                         nnzA,
                                         descrA,
                                         dptrA,
                                         dindA,
                                         dvalA,
                                         nnzT,
                                         dptrT,
                                         dindT,
                                         dvalT,
                                         dB,
                                         tolerance,
                                         reorder,
                                         dX,
                                         hptrA,
                                         hindA,
                                         hvalA,
                                         hptrT,
                                         hptrTRes,
                                         hindT,
                                         hindTRes,
                                         hvalT,
                                         hvalTRes,
                                         hB,
                                         hX,
                                         hXRes,
                                         hSingularity,
                                         &max_error,
                                         testcase);

        // collect performance data
        if(argus.timing)
            csrlsvchol_getPerfData<HOST, T>(handle,
                                            n,
                                            nnzA,
                                            descrA,
                                            dptrA,
                                            dindA,
                                            dvalA,
                                            nnzT,
                                            dptrT,
                                            dindT,
                                            dvalT,
                                            dB,
                                            tolerance,
                                            reorder,
                                            dX,
                                            hptrA,
                                            hindA,
                                            hvalA,
                                            hptrT,
                                            hindT,
                                            hvalT,
                                            hB,
                                            hX,
                                            hSingularity,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf,
                                            testcase);
    }

    else
    {
        // memory allocations
        device_strided_batch_vector<int> dptrA(size_ptrA, 1, size_ptrA, 1);
        device_strided_batch_vector<int> dindA(size_indA, 1, size_indA, 1);
        device_strided_batch_vector<T>   dvalA(size_valA, 1, size_valA, 1);
        device_strided_batch_vector<int> dptrT(size_ptrT, 1, size_ptrT, 1);
        device_strided_batch_vector<int> dindT(size_indT, 1, size_indT, 1);
        device_strided_batch_vector<T>   dvalT(size_valT, 1, size_valT, 1);
        device_strided_batch_vector<T>   dB(size_BX, 1, size_BX, 1);
        device_strided_batch_vector<T>   dX(size_BX, 1, size_BX, 1);
        CHECK_HIP_ERROR(dptrA.memcheck());
        CHECK_HIP_ERROR(dptrT.memcheck());
        if(size_indA)
            CHECK_HIP_ERROR(dindA.memcheck());
        if(size_valA)
            CHECK_HIP_ERROR(dvalA.memcheck());
        if(size_indT)
            CHECK_HIP_ERROR(dindT.memcheck());
        if(size_valT)
            CHECK_HIP_ERROR(dvalT.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dB.memcheck());
        if(size_BX)
            CHECK_HIP_ERROR(dX.memcheck());

        // check computations
        if(argus.unit_check || argus.norm_check)
            csrlsvchol_getError<HOST, T>(handle,
                                         n,
                                         nnzA,
                                         descrA,
                                         dptrA,
                                         dindA,
                                         dvalA,
                                         nnzT,
                                         dptrT,
                                         dindT,
                                         dvalT,
                                         dB,
                                         tolerance,
                                         reorder,
                                         dX,
                                         hptrA,
                                         hindA,
                                         hvalA,
                                         hptrT,
                                         hptrTRes,
                                         hindT,
                                         hindTRes,
                                         hvalT,
                                         hvalTRes,
                                         hB,
                                         hX,
                                         hXRes,
                                         hSingularity,
                                         &max_error,
                                         testcase);

        // collect performance data
        if(argus.timing)
            csrlsvchol_getPerfData<HOST, T>(handle,
                                            n,
                                            nnzA,
                                            descrA,
                                            dptrA,
                                            dindA,
                                            dvalA,
                                            nnzT,
                                            dptrT,
                                            dindT,
                                            dvalT,
                                            dB,
                                            tolerance,
                                            reorder,
                                            dX,
                                            hptrA,
                                            hindA,
                                            hvalA,
                                            hptrT,
                                            hindT,
                                            hvalT,
                                            hB,
                                            hX,
                                            hSingularity,
                                            &gpu_time_used,
                                            &cpu_time_used,
                                            hot_calls,
                                            argus.perf,
                                            testcase);
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            std::cerr << "\n============================================\n";
            std::cerr << "Arguments:\n";
            std::cerr << "============================================\n";
            rocsolver_bench_output("n", "nnzA");
            rocsolver_bench_output(n, nnzA);

            std::cerr << "\n============================================\n";
            std::cerr << "Results:\n";
            std::cerr << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
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
