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

/******************** HANDLE ********************/
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

    uint8_t *d_buffer, *h_buffer;

    // Constructor
    explicit hipsolverRfHandle()
        : fast_mode(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
        , matrix_format(HIPSOLVERRF_MATRIX_FORMAT_CSR)
        , diag_format(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)
        , numeric_boost(HIPSOLVERRF_NUMERIC_BOOST_NOT_USED)
        , fact_alg(HIPSOLVERRF_FACTORIZATION_ALG0)
        , solve_alg(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)
        , n(0)
        , nnzA(0)
        , nnzL(0)
        , nnzU(0)
        , nnzLU(0)
        , batch_count(0)
        , effective_zero(0.0)
        , boost_val(0.0)
        , d_buffer(nullptr)
        , h_buffer(nullptr)
    {
    }

    // Allocate device memory
    hipsolverStatus_t malloc_device(int n, int nnzA, int nnzL, int nnzU)
    {
        if(this->n != n || this->nnzA != nnzA || this->nnzL != nnzL || this->nnzU != nnzU)
        {
            int nnzLU = nnzL - n + nnzU;

            if(this->h_buffer)
            {
                free(this->h_buffer);
                this->h_buffer = nullptr;
            }

            if(this->d_buffer)
            {
                if(hipFree(this->d_buffer) != hipSuccess)
                    return HIPSOLVER_STATUS_INTERNAL_ERROR;
                this->d_buffer = nullptr;
            }

            size_t size_dPtrA = sizeof(rocblas_int) * n;
            size_t size_dIndA = sizeof(rocblas_int) * nnzA;
            size_t size_dValA = sizeof(double) * nnzA;

            size_t size_dPtrL = sizeof(rocblas_int) * n;
            size_t size_dIndL = sizeof(rocblas_int) * nnzL;
            size_t size_dValL = sizeof(double) * nnzL;

            size_t size_dPtrU = sizeof(rocblas_int) * n;
            size_t size_dIndU = sizeof(rocblas_int) * nnzU;
            size_t size_dValU = sizeof(double) * nnzU;

            size_t size_dPtrLU = sizeof(rocblas_int) * n;
            size_t size_dIndLU = sizeof(rocblas_int) * nnzLU;
            size_t size_dValLU = sizeof(double) * nnzLU;

            size_t size_dP = sizeof(rocblas_int) * n;
            size_t size_dQ = sizeof(rocblas_int) * n;

            size_t size_buffer = size_dPtrA + size_dIndA + size_dValA + size_dPtrL + size_dIndL
                                 + size_dValL + size_dPtrU + size_dIndU + size_dValU + size_dPtrLU
                                 + size_dIndLU + size_dValLU + size_dP + size_dQ;

            if(hipMalloc(&this->d_buffer, size_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            uint8_t* temp_buf;
            this->dPtrA  = (rocblas_int*)(temp_buf = this->d_buffer);
            this->dIndA  = (rocblas_int*)(temp_buf += size_dPtrA);
            this->dValA  = (double*)(temp_buf += size_dIndA);
            this->dPtrL  = (rocblas_int*)(temp_buf += size_dValA);
            this->dIndL  = (rocblas_int*)(temp_buf += size_dPtrL);
            this->dValL  = (double*)(temp_buf += size_dIndL);
            this->dPtrU  = (rocblas_int*)(temp_buf += size_dValL);
            this->dIndU  = (rocblas_int*)(temp_buf += size_dPtrU);
            this->dValU  = (double*)(temp_buf += size_dIndU);
            this->dPtrLU = (rocblas_int*)(temp_buf += size_dValU);
            this->dIndLU = (rocblas_int*)(temp_buf += size_dPtrLU);
            this->dValLU = (double*)(temp_buf += size_dIndLU);
            this->dP     = (rocblas_int*)(temp_buf += size_dValLU);
            this->dQ     = (rocblas_int*)(temp_buf += size_dP);

            this->n     = n;
            this->nnzA  = nnzA;
            this->nnzL  = nnzL;
            this->nnzU  = nnzU;
            this->nnzLU = nnzLU;
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Allocate host memory
    hipsolverStatus_t malloc_host()
    {
        if(!this->h_buffer)
        {
            size_t size_hPtrL = sizeof(rocblas_int) * n;
            size_t size_hIndL = sizeof(rocblas_int) * nnzL;
            size_t size_hValL = sizeof(double) * nnzL;

            size_t size_hPtrU = sizeof(rocblas_int) * n;
            size_t size_hIndU = sizeof(rocblas_int) * nnzU;
            size_t size_hValU = sizeof(double) * nnzU;

            size_t size_hPtrLU = sizeof(rocblas_int) * n;
            size_t size_hIndLU = sizeof(rocblas_int) * nnzLU;
            size_t size_hValLU = sizeof(double) * nnzLU;

            size_t size_buffer = std::max(size_hPtrL + size_hIndL + size_hValL + size_hPtrU
                                              + size_hIndU + size_hValU,
                                          size_hPtrLU + size_hIndLU + size_hValLU);

            this->h_buffer = (uint8_t*)malloc(size_buffer);
            if(!this->h_buffer)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            uint8_t* temp_buf;
            this->hPtrL = (rocblas_int*)(temp_buf = this->h_buffer);
            this->hIndL = (rocblas_int*)(temp_buf += size_hPtrL);
            this->hValL = (double*)(temp_buf += size_hIndL);
            this->hPtrU = (rocblas_int*)(temp_buf += size_hValL);
            this->hIndU = (rocblas_int*)(temp_buf += size_hPtrU);
            this->hValU = (double*)(temp_buf += size_hIndU);

            this->hPtrLU = (rocblas_int*)(temp_buf = this->h_buffer);
            this->hIndLU = (rocblas_int*)(temp_buf += size_hPtrLU);
            this->hValLU = (double*)(temp_buf += size_hIndLU);
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Free memory
    void free_all()
    {
        if(this->h_buffer)
        {
            free(this->h_buffer);
            this->h_buffer = nullptr;
        }

        if(this->d_buffer)
        {
            hipFree(this->d_buffer);
            this->d_buffer = nullptr;
        }
    }
};

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
