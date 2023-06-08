/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipsolver.h"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

struct hipsolverRfHandle
{
    hipsolverRfResetValuesFastMode_t fast_mode;
    hipsolverRfMatrixFormat_t        matrix_format;
    hipsolverRfUnitDiagonal_t        diag_format;
    hipsolverRfNumericBoostReport_t  numeric_boost;

    hipsolverRfFactorization_t   fact_alg;
    hipsolverRfTriangularSolve_t solve_alg;

    rocblas_handle   handle;
    rocsolver_rfinfo rfinfo;

    rocblas_int n, nnzA, nnzL, nnzU, nnzLU, batch_count;
    double      effective_zero;
    double      boost_val;

    rocblas_int* dPtrA;
    rocblas_int* dIndA;
    double*      dValA;

    rocblas_int *dPtrL, *hPtrL;
    rocblas_int *dIndL, *hIndL;
    double *     dValL, *hValL;

    rocblas_int *dPtrU, *hPtrU;
    rocblas_int *dIndU, *hIndU;
    double *     dValU, *hValU;

    rocblas_int *dPtrLU, *hPtrLU;
    rocblas_int *dIndLU, *hIndLU;
    double *     dValLU, *hValLU;

    rocblas_int* dP;
    rocblas_int* dQ;

    char *d_buffer, *h_buffer;

    // Constructor
    explicit hipsolverRfHandle();

    hipsolverRfHandle(const hipsolverRfHandle&) = delete;

    hipsolverRfHandle(hipsolverRfHandle&&) = delete;

    hipsolverRfHandle& operator=(const hipsolverRfHandle&) = delete;

    hipsolverRfHandle& operator=(hipsolverRfHandle&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup();
    hipsolverStatus_t teardown();

    hipsolverStatus_t malloc_device(int n, int nnzA, int nnzL, int nnzU);
    hipsolverStatus_t malloc_host();
    hipsolverStatus_t free_mem();
};

struct hipsolverGesvdjInfo
{
    int     capacity;
    int     batch_count;
    int*    n_sweeps;
    double* residual;

    int    max_sweeps;
    double tolerance;
    bool   is_batched, is_float, sort_eig;

    char* d_buffer;

    // Constructor
    hipsolverGesvdjInfo();

    hipsolverGesvdjInfo(const hipsolverGesvdjInfo&) = delete;

    hipsolverGesvdjInfo(hipsolverGesvdjInfo&&) = delete;

    hipsolverGesvdjInfo& operator=(const hipsolverGesvdjInfo&) = delete;

    hipsolverGesvdjInfo& operator=(hipsolverGesvdjInfo&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup(int bc);
    hipsolverStatus_t teardown();
};

struct hipsolverSyevjInfo
{
    int     capacity;
    int     batch_count;
    int*    n_sweeps;
    double* residual;

    int    max_sweeps;
    double tolerance;
    bool   is_batched, is_float, sort_eig;

    char* d_buffer;

    // Constructor
    hipsolverSyevjInfo();

    hipsolverSyevjInfo(const hipsolverSyevjInfo&) = delete;

    hipsolverSyevjInfo(hipsolverSyevjInfo&&) = delete;

    hipsolverSyevjInfo& operator=(const hipsolverSyevjInfo&) = delete;

    hipsolverSyevjInfo& operator=(hipsolverSyevjInfo&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup(int bc);
    hipsolverStatus_t teardown();
};
