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
#include "hipsolver_types.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

extern "C" {

/******************** HANDLE ********************/
hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    *handle                  = new hipsolverRfHandle;
    hipsolverStatus_t result = (*handle)->setup();

    if(result != HIPSOLVER_STATUS_SUCCESS)
    {
        delete *handle;
        *handle = nullptr;
    }

    return result;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverStatus_t result = handle->teardown();
    delete handle;

    return result;
}
catch(...)
{
    return exception2hip_status();
}

/******************** NON-BATCHED ********************/
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrL || !csrColIndL || !csrValL)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrU || !csrColIndU || !csrValU)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    CHECK_HIPSOLVER_ERROR(handle->malloc_device(n, nnzA, nnzL, nnzU));

    CHECK_HIP_ERROR(hipMemcpy(
        handle->dPtrA, csrRowPtrA, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dIndA, csrColIndA, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyDeviceToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_sumlu(handle->handle,
                                               n,
                                               nnzL,
                                               csrRowPtrL,
                                               csrColIndL,
                                               csrValL,
                                               nnzU,
                                               csrRowPtrU,
                                               csrColIndU,
                                               csrValU,
                                               handle->dPtrLU,
                                               handle->dIndLU,
                                               handle->dValLU));

    CHECK_HIP_ERROR(hipMemcpy(handle->dP, P, sizeof(rocblas_int) * n, hipMemcpyDeviceToDevice));
    CHECK_HIP_ERROR(hipMemcpy(handle->dQ, Q, sizeof(rocblas_int) * n, hipMemcpyDeviceToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfSetupHost(int                 n,
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrL || !csrColIndL || !csrValL)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtrU || !csrColIndU || !csrValU)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    CHECK_HIPSOLVER_ERROR(handle->malloc_device(n, nnzA, nnzL, nnzU));

    CHECK_HIP_ERROR(
        hipMemcpy(handle->dPtrA, csrRowPtrA, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dIndA, csrColIndA, sizeof(rocblas_int) * nnzA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(
        hipMemcpy(handle->dPtrL, csrRowPtrL, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dIndL, csrColIndL, sizeof(rocblas_int) * nnzL, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dValL, csrValL, sizeof(double) * nnzL, hipMemcpyHostToDevice));

    CHECK_HIP_ERROR(
        hipMemcpy(handle->dPtrU, csrRowPtrU, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dIndU, csrColIndU, sizeof(rocblas_int) * nnzU, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(handle->dValU, csrValU, sizeof(double) * nnzU, hipMemcpyHostToDevice));

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_sumlu(handle->handle,
                                               n,
                                               nnzL,
                                               handle->dPtrL,
                                               handle->dIndL,
                                               handle->dValL,
                                               nnzU,
                                               handle->dPtrU,
                                               handle->dIndU,
                                               handle->dValU,
                                               handle->dPtrLU,
                                               handle->dIndLU,
                                               handle->dValLU));

    CHECK_HIP_ERROR(hipMemcpy(handle->dP, P, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(handle->dQ, Q, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfAccessBundledFactorsDevice(
    hipsolverRfHandle_t handle, int* nnzM, int** Mp, int** Mi, double** Mx)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!nnzM || !Mp || !Mi || !Mx)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *nnzM = handle->nnzLU;
    *Mp   = handle->dPtrLU;
    *Mi   = handle->dIndLU;
    *Mx   = handle->dValLU;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return rocblas2hip_status(rocsolver_dcsrrf_analysis(handle->handle,
                                                        handle->n,
                                                        1,
                                                        handle->nnzA,
                                                        handle->dPtrA,
                                                        handle->dIndA,
                                                        handle->dValA,
                                                        handle->nnzLU,
                                                        handle->dPtrLU,
                                                        handle->dIndLU,
                                                        handle->dValLU,
                                                        handle->dP,
                                                        handle->dQ,
                                                        // pass dummy values for B
                                                        handle->dValA,
                                                        handle->n,
                                                        handle->rfinfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(
    hipsolverRfHandle_t handle, int* h_nnzM, int** h_Mp, int** h_Mi, double** h_Mx)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!h_nnzM || !h_Mp || !h_Mi || !h_Mx)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    CHECK_HIPSOLVER_ERROR(handle->malloc_host());

    CHECK_HIP_ERROR(hipMemcpy(handle->hPtrLU,
                              handle->dPtrLU,
                              sizeof(rocblas_int) * (handle->n + 1),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(handle->hIndLU,
                              handle->dIndLU,
                              sizeof(rocblas_int) * handle->nnzLU,
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        handle->hValLU, handle->dValLU, sizeof(double) * handle->nnzLU, hipMemcpyDeviceToHost));

    *h_nnzM = handle->nnzLU;
    *h_Mp   = handle->hPtrLU;
    *h_Mi   = handle->hIndLU;
    *h_Mx   = handle->hValLU;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!h_nnzL || !h_Lp || !h_Li || !h_Lx)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!h_nnzU || !h_Up || !h_Ui || !h_Ux)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    CHECK_HIPSOLVER_ERROR(handle->malloc_host());

    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_splitlu(handle->handle,
                                                 handle->n,
                                                 handle->nnzLU,
                                                 handle->dPtrLU,
                                                 handle->dIndLU,
                                                 handle->dValLU,
                                                 handle->dPtrL,
                                                 handle->dIndL,
                                                 handle->dValL,
                                                 handle->dPtrU,
                                                 handle->dIndU,
                                                 handle->dValU));

    CHECK_HIP_ERROR(hipMemcpy(handle->hPtrL,
                              handle->dPtrL,
                              sizeof(rocblas_int) * (handle->n + 1),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        handle->hIndL, handle->dIndL, sizeof(rocblas_int) * handle->nnzL, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        handle->hValL, handle->dValL, sizeof(double) * handle->nnzL, hipMemcpyDeviceToHost));

    CHECK_HIP_ERROR(hipMemcpy(handle->hPtrU,
                              handle->dPtrU,
                              sizeof(rocblas_int) * (handle->n + 1),
                              hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        handle->hIndU, handle->dIndU, sizeof(rocblas_int) * handle->nnzU, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        handle->hValU, handle->dValU, sizeof(double) * handle->nnzU, hipMemcpyDeviceToHost));

    *h_nnzL = handle->nnzL;
    *h_Lp   = handle->hPtrL;
    *h_Li   = handle->hIndL;
    *h_Lx   = handle->hValL;

    *h_nnzU = handle->nnzU;
    *h_Up   = handle->hPtrU;
    *h_Ui   = handle->hIndU;
    *h_Ux   = handle->hValU;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!fact_alg)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!solve_alg)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *fact_alg  = handle->fact_alg;
    *solve_alg = handle->solve_alg;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!format)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!diag)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *format = handle->matrix_format;
    *diag   = handle->diag_format;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t              handle,
                                                   hipsolverRfNumericBoostReport_t* report)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!report)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *report = handle->numeric_boost;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t
    hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double* zero, double* boost)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!zero)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!boost)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *zero  = handle->effective_zero;
    *boost = handle->boost_val;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t               handle,
                                                    hipsolverRfResetValuesFastMode_t* fastMode)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!fastMode)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *fastMode = handle->fast_mode;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return rocblas2hip_status(rocsolver_dcsrrf_refactlu(handle->handle,
                                                        handle->n,
                                                        handle->nnzA,
                                                        handle->dPtrA,
                                                        handle->dIndA,
                                                        handle->dValA,
                                                        handle->nnzLU,
                                                        handle->dPtrLU,
                                                        handle->dIndLU,
                                                        handle->dValLU,
                                                        handle->dP,
                                                        handle->dQ,
                                                        handle->rfinfo));
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!csrRowPtrA || !csrColIndA || !csrValA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!P || !Q)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(handle->n != n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(handle->nnzA != nnzA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    CHECK_HIP_ERROR(
        hipMemcpy(handle->dValA, csrValA, sizeof(double) * nnzA, hipMemcpyDeviceToDevice));

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!handle->d_buffer)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return rocblas2hip_status(rocsolver_dcsrrf_solve(handle->handle,
                                                     handle->n,
                                                     nrhs,
                                                     handle->nnzLU,
                                                     handle->dPtrLU,
                                                     handle->dIndLU,
                                                     handle->dValLU,
                                                     handle->dP,
                                                     handle->dQ,
                                                     XF,
                                                     ldxf,
                                                     handle->rfinfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** BATCHED ********************/
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
