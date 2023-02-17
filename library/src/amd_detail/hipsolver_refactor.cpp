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

/*! \file
 *  \brief Implementation of the compatibility refactor APIs that require especial
 *  calls to hipSOLVER or rocSOLVER.
 */

#include "error_macros.hpp"
#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

extern "C" {

hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

// non-batched routines
hipsolverStatus_t hipsolverRfSetupDevice(int                 n,
                                         int                 nnzA,
                                         int*                csrRowPtrA,
                                         int*                csrColIndA,
                                         double*             csrValA,
                                         int                 nnzL,
                                         int*                csrRowPtrL,
                                         int*                csrColIndL,
                                         double*             csrValL,
                                         int                 nnzU,
                                         int*                csrRowPtrU,
                                         int*                csrColIndU,
                                         double*             csrValU,
                                         int*                P,
                                         int*                Q,
                                         hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetupHost(int                 n,
                                       int                 nnzA,
                                       int*                h_csrRowPtrA,
                                       int*                h_csrColIndA,
                                       double*             h_csrValA,
                                       int                 nnzL,
                                       int*                h_csrRowPtrL,
                                       int*                h_csrColIndL,
                                       double*             h_csrValL,
                                       int                 nnzU,
                                       int*                h_csrRowPtrU,
                                       int*                h_csrColIndU,
                                       double*             h_csrValU,
                                       int*                h_P,
                                       int*                h_Q,
                                       hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(
    hipsolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(
    hipsolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle,
                                                     int*                h_nnzL,
                                                     int**               h_Lp,
                                                     int**               h_Li,
                                                     double**            h_Lx,
                                                     int*                h_nnzU,
                                                     int**               h_Up,
                                                     int**               h_Ui,
                                                     double**            h_Ux)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGet_Algs(hipsolverRfHandle_t           handle,
                                      hipsolverRfFactorization_t*   fact_alg,
                                      hipsolverRfTriangularSolve_t* solve_alg)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t        handle,
                                             hipsolverRfMatrixFormat_t* format,
                                             hipsolverRfUnitDiagonal_t* diag)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t              handle,
                                                   hipsolverRfNumericBoostReport_t* report)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t
    hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double* zero, double* boost)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t               handle,
                                                    hipsolverRfResetValuesFastMode_t* fastMode)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfResetValues(int                 n,
                                         int                 nnzA,
                                         int*                csrRowPtrA,
                                         int*                csrColIndA,
                                         double*             csrValA,
                                         int*                P,
                                         int*                Q,
                                         hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t          handle,
                                     hipsolverRfFactorization_t   fact_alg,
                                     hipsolverRfTriangularSolve_t solve_alg)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t       handle,
                                             hipsolverRfMatrixFormat_t format,
                                             hipsolverRfUnitDiagonal_t diag)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle,
                                                  double              effective_zero,
                                                  double              boost_val)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t              handle,
                                                    hipsolverRfResetValuesFastMode_t fastMode)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle,
                                   int*                P,
                                   int*                Q,
                                   int                 nrhs,
                                   double*             Temp,
                                   int                 ldt,
                                   double*             XF,
                                   int                 ldxf)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

// batched routines
hipsolverStatus_t hipsolverRfBatchSetupHost(int                 batchSize,
                                            int                 n,
                                            int                 nnzA,
                                            int*                h_csrRowPtrA,
                                            int*                h_csrColIndA,
                                            double*             h_csrValA_array[],
                                            int                 nnzL,
                                            int*                h_csrRowPtrL,
                                            int*                h_csrColIndL,
                                            double*             h_csrValL,
                                            int                 nnzU,
                                            int*                h_csrRowPtrU,
                                            int*                h_csrColIndU,
                                            double*             h_csrValU,
                                            int*                h_P,
                                            int*                h_Q,
                                            hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchResetValues(int                 batchSize,
                                              int                 n,
                                              int                 nnzA,
                                              int*                csrRowPtrA,
                                              int*                csrColIndA,
                                              double*             csrValA_array[],
                                              int*                P,
                                              int*                Q,
                                              hipsolverRfHandle_t handle)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle,
                                        int*                P,
                                        int*                Q,
                                        int                 nrhs,
                                        double*             Temp,
                                        int                 ldt,
                                        double*             XF_array[],
                                        int                 ldxf)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle, int* position)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

} //extern C
