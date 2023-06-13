/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief Implementation of the hipSOLVER regular APIs on the cuSOLVER side.
 */

#include "hipsolver.h"
#include "exceptions.hpp"
#include "hipsolver_conversions.hpp"
#include "hipsolver_types.hpp"

#include <cusolverDn.h>

extern "C" {

/******************** AUXILIARY ********************/
hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    *handle                  = new hipsolverHandle;
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

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
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

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSetStream(handle->handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnGetStream(handle->handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVDJ PARAMS ********************/
hipsolverStatus_t hipsolverCreateGesvdjInfo(hipsolverGesvdjInfo_t* info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *info                    = new hipsolverGesvdjInfo;
    hipsolverStatus_t result = (*info)->setup();

    if(result != HIPSOLVER_STATUS_SUCCESS)
    {
        delete *info;
        *info = nullptr;
    }

    return result;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroyGesvdjInfo(hipsolverGesvdjInfo_t info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverStatus_t result = info->teardown();
    delete info;

    return result;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info, int max_sweeps)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXgesvdjSetMaxSweeps(info->info, max_sweeps));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetSortEig(hipsolverGesvdjInfo_t info, int sort_eig)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXgesvdjSetSortEig(info->info, sort_eig));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjSetTolerance(hipsolverGesvdjInfo_t info, double tolerance)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXgesvdjSetTolerance(info->info, tolerance));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjGetResidual(hipsolverDnHandle_t   handle,
                                              hipsolverGesvdjInfo_t info,
                                              double*               residual)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXgesvdjGetResidual(handle->handle, info->info, residual));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXgesvdjGetSweeps(hipsolverDnHandle_t   handle,
                                            hipsolverGesvdjInfo_t info,
                                            int*                  executed_sweeps)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXgesvdjGetSweeps(handle->handle, info->info, executed_sweeps));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYEVJ PARAMS ********************/
hipsolverStatus_t hipsolverCreateSyevjInfo(hipsolverSyevjInfo_t* info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *info                    = new hipsolverSyevjInfo;
    hipsolverStatus_t result = (*info)->setup();

    if(result != HIPSOLVER_STATUS_SUCCESS)
    {
        delete *info;
        *info = nullptr;
    }

    return result;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroySyevjInfo(hipsolverSyevjInfo_t info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverStatus_t result = info->teardown();
    delete info;

    return result;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info, int max_sweeps)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXsyevjSetMaxSweeps(info->info, max_sweeps));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetSortEig(hipsolverSyevjInfo_t info, int sort_eig)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXsyevjSetSortEig(info->info, sort_eig));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjSetTolerance(hipsolverSyevjInfo_t info, double tolerance)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXsyevjSetTolerance(info->info, tolerance));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjGetResidual(hipsolverDnHandle_t  handle,
                                             hipsolverSyevjInfo_t info,
                                             double*              residual)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXsyevjGetResidual(handle->handle, info->info, residual));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverXsyevjGetSweeps(hipsolverDnHandle_t  handle,
                                           hipsolverSyevjInfo_t info,
                                           int*                 executed_sweeps)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnXsyevjGetSweeps(handle->handle, info->info, executed_sweeps));
}
catch(...)
{
    return exception2hip_status();
}

/******************** ORGBR/UNGBR ********************/
hipsolverStatus_t hipsolverSorgbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             float*              A,
                                             int                 lda,
                                             float*              tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSorgbr_bufferSize(
        handle->handle, hip2cuda_side(side), m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             double*             A,
                                             int                 lda,
                                             double*             tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDorgbr_bufferSize(
        handle->handle, hip2cuda_side(side), m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             hipFloatComplex*    tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungbr_bufferSize(
        handle->handle, hip2cuda_side(side), m, n, k, (cuComplex*)A, lda, (cuComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungbr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverSideMode_t side,
                                             int                 m,
                                             int                 n,
                                             int                 k,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             hipDoubleComplex*   tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungbr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       m,
                                                       n,
                                                       k,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)tau,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgbr(hipsolverHandle_t   handle,
                                  hipsolverSideMode_t side,
                                  int                 m,
                                  int                 n,
                                  int                 k,
                                  float*              A,
                                  int                 lda,
                                  float*              tau,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSorgbr(
        handle->handle, hip2cuda_side(side), m, n, k, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgbr(hipsolverHandle_t   handle,
                                  hipsolverSideMode_t side,
                                  int                 m,
                                  int                 n,
                                  int                 k,
                                  double*             A,
                                  int                 lda,
                                  double*             tau,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDorgbr(
        handle->handle, hip2cuda_side(side), m, n, k, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungbr(hipsolverHandle_t   handle,
                                  hipsolverSideMode_t side,
                                  int                 m,
                                  int                 n,
                                  int                 k,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    tau,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungbr(handle->handle,
                                            hip2cuda_side(side),
                                            m,
                                            n,
                                            k,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungbr(hipsolverHandle_t   handle,
                                  hipsolverSideMode_t side,
                                  int                 m,
                                  int                 n,
                                  int                 k,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   tau,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungbr(handle->handle,
                                            hip2cuda_side(side),
                                            m,
                                            n,
                                            k,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** ORGQR/UNGQR ********************/
hipsolverStatus_t hipsolverSorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSorgqr_bufferSize(handle->handle, m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDorgqr_bufferSize(handle->handle, m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               k,
                                             hipFloatComplex*  A,
                                             int               lda,
                                             hipFloatComplex*  tau,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungqr_bufferSize(
        handle->handle, m, n, k, (cuComplex*)A, lda, (cuComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               k,
                                             hipDoubleComplex* A,
                                             int               lda,
                                             hipDoubleComplex* tau,
                                             int*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungqr_bufferSize(
        handle->handle, m, n, k, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  float*            A,
                                  int               lda,
                                  float*            tau,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSorgqr(handle->handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  double*           A,
                                  int               lda,
                                  double*           tau,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDorgqr(handle->handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  tau,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungqr(handle->handle,
                                            m,
                                            n,
                                            k,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* tau,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungqr(handle->handle,
                                            m,
                                            n,
                                            k,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** ORGTR/UNGTR ********************/
hipsolverStatus_t hipsolverSorgtr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A,
                                             int                 lda,
                                             float*              tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSorgtr_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgtr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A,
                                             int                 lda,
                                             double*             tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDorgtr_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungtr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             hipFloatComplex*    tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungtr_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuComplex*)A, lda, (cuComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungtr_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             hipDoubleComplex*   tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungtr_bufferSize(handle->handle,
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)tau,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSorgtr(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              tau,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSorgtr(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgtr(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             tau,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDorgtr(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungtr(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    tau,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCungtr(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungtr(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   tau,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZungtr(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** ORMQR/UNMQR ********************/
hipsolverStatus_t hipsolverSormqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             float*               A,
                                             int                  lda,
                                             float*               tau,
                                             float*               C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSormqr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       k,
                                                       A,
                                                       lda,
                                                       tau,
                                                       C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDormqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             double*              A,
                                             int                  lda,
                                             double*              tau,
                                             double*              C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDormqr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       k,
                                                       A,
                                                       lda,
                                                       tau,
                                                       C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             hipFloatComplex*     tau,
                                             hipFloatComplex*     C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCunmqr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       k,
                                                       (cuComplex*)A,
                                                       lda,
                                                       (cuComplex*)tau,
                                                       (cuComplex*)C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             int                  k,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             hipDoubleComplex*    tau,
                                             hipDoubleComplex*    C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZunmqr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       k,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)tau,
                                                       (cuDoubleComplex*)C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSormqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  float*               A,
                                  int                  lda,
                                  float*               tau,
                                  float*               C,
                                  int                  ldc,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSormqr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            k,
                                            A,
                                            lda,
                                            tau,
                                            C,
                                            ldc,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDormqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  double*              A,
                                  int                  lda,
                                  double*              tau,
                                  double*              C,
                                  int                  ldc,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDormqr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            k,
                                            A,
                                            lda,
                                            tau,
                                            C,
                                            ldc,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  hipFloatComplex*     tau,
                                  hipFloatComplex*     C,
                                  int                  ldc,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCunmqr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            k,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)C,
                                            ldc,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  hipDoubleComplex*    tau,
                                  hipDoubleComplex*    C,
                                  int                  ldc,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZunmqr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            k,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)C,
                                            ldc,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** ORMTR/UNMTR ********************/
hipsolverStatus_t hipsolverSormtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             float*               A,
                                             int                  lda,
                                             float*               tau,
                                             float*               C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSormtr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_fill(uplo),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       A,
                                                       lda,
                                                       tau,
                                                       C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDormtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             double*              A,
                                             int                  lda,
                                             double*              tau,
                                             double*              C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDormtr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_fill(uplo),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       A,
                                                       lda,
                                                       tau,
                                                       C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             hipFloatComplex*     tau,
                                             hipFloatComplex*     C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCunmtr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_fill(uplo),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       (cuComplex*)tau,
                                                       (cuComplex*)C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmtr_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverSideMode_t  side,
                                             hipsolverFillMode_t  uplo,
                                             hipsolverOperation_t trans,
                                             int                  m,
                                             int                  n,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             hipDoubleComplex*    tau,
                                             hipDoubleComplex*    C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZunmtr_bufferSize(handle->handle,
                                                       hip2cuda_side(side),
                                                       hip2cuda_fill(uplo),
                                                       hip2cuda_operation(trans),
                                                       m,
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)tau,
                                                       (cuDoubleComplex*)C,
                                                       ldc,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSormtr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverFillMode_t  uplo,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  float*               A,
                                  int                  lda,
                                  float*               tau,
                                  float*               C,
                                  int                  ldc,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSormtr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_fill(uplo),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            A,
                                            lda,
                                            tau,
                                            C,
                                            ldc,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDormtr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverFillMode_t  uplo,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  double*              A,
                                  int                  lda,
                                  double*              tau,
                                  double*              C,
                                  int                  ldc,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDormtr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_fill(uplo),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            A,
                                            lda,
                                            tau,
                                            C,
                                            ldc,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCunmtr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverFillMode_t  uplo,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  hipFloatComplex*     tau,
                                  hipFloatComplex*     C,
                                  int                  ldc,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCunmtr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_fill(uplo),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)C,
                                            ldc,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZunmtr(hipsolverHandle_t    handle,
                                  hipsolverSideMode_t  side,
                                  hipsolverFillMode_t  uplo,
                                  hipsolverOperation_t trans,
                                  int                  m,
                                  int                  n,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  hipDoubleComplex*    tau,
                                  hipDoubleComplex*    C,
                                  int                  ldc,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZunmtr(handle->handle,
                                            hip2cuda_side(side),
                                            hip2cuda_fill(uplo),
                                            hip2cuda_operation(trans),
                                            m,
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)C,
                                            ldc,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GEBRD ********************/
hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgebrd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgebrd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgebrd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgebrd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgebrd(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            D,
                                  float*            E,
                                  float*            tauq,
                                  float*            taup,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSgebrd(handle->handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgebrd(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           D,
                                  double*           E,
                                  double*           tauq,
                                  double*           taup,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDgebrd(handle->handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  float*            D,
                                  float*            E,
                                  hipFloatComplex*  tauq,
                                  hipFloatComplex*  taup,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgebrd(handle->handle,
                                            m,
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            D,
                                            E,
                                            (cuComplex*)tauq,
                                            (cuComplex*)taup,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  double*           D,
                                  double*           E,
                                  hipDoubleComplex* tauq,
                                  hipDoubleComplex* taup,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgebrd(handle->handle,
                                            m,
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            D,
                                            E,
                                            (cuDoubleComplex*)tauq,
                                            (cuDoubleComplex*)taup,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GELS ********************/
hipsolverStatus_t hipsolverSSgels_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               nrhs,
                                             float*            A,
                                             int               lda,
                                             float*            B,
                                             int               ldb,
                                             float*            X,
                                             int               ldx,
                                             size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSSgels_bufferSize(
        handle->handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgels_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               nrhs,
                                             double*           A,
                                             int               lda,
                                             double*           B,
                                             int               ldb,
                                             double*           X,
                                             int               ldx,
                                             size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDDgels_bufferSize(
        handle->handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgels_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               nrhs,
                                             hipFloatComplex*  A,
                                             int               lda,
                                             hipFloatComplex*  B,
                                             int               ldb,
                                             hipFloatComplex*  X,
                                             int               ldx,
                                             size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCCgels_bufferSize(handle->handle,
                                                       m,
                                                       n,
                                                       nrhs,
                                                       (cuComplex*)A,
                                                       lda,
                                                       (cuComplex*)B,
                                                       ldb,
                                                       (cuComplex*)X,
                                                       ldx,
                                                       nullptr,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgels_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               nrhs,
                                             hipDoubleComplex* A,
                                             int               lda,
                                             hipDoubleComplex* B,
                                             int               ldb,
                                             hipDoubleComplex* X,
                                             int               ldx,
                                             size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZZgels_bufferSize(handle->handle,
                                                       m,
                                                       n,
                                                       nrhs,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)B,
                                                       ldb,
                                                       (cuDoubleComplex*)X,
                                                       ldx,
                                                       nullptr,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSSgels(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               nrhs,
                                  float*            A,
                                  int               lda,
                                  float*            B,
                                  int               ldb,
                                  float*            X,
                                  int               ldx,
                                  void*             work,
                                  size_t            lwork,
                                  int*              niters,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSSgels(
        handle->handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDDgels(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               nrhs,
                                  double*           A,
                                  int               lda,
                                  double*           B,
                                  int               ldb,
                                  double*           X,
                                  int               ldx,
                                  void*             work,
                                  size_t            lwork,
                                  int*              niters,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDDgels(
        handle->handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCCgels(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               nrhs,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  B,
                                  int               ldb,
                                  hipFloatComplex*  X,
                                  int               ldx,
                                  void*             work,
                                  size_t            lwork,
                                  int*              niters,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCCgels(handle->handle,
                                            m,
                                            n,
                                            nrhs,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)B,
                                            ldb,
                                            (cuComplex*)X,
                                            ldx,
                                            work,
                                            lwork,
                                            niters,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZZgels(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               nrhs,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* B,
                                  int               ldb,
                                  hipDoubleComplex* X,
                                  int               ldx,
                                  void*             work,
                                  size_t            lwork,
                                  int*              niters,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZZgels(handle->handle,
                                            m,
                                            n,
                                            nrhs,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            (cuDoubleComplex*)X,
                                            ldx,
                                            work,
                                            lwork,
                                            niters,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GEQRF ********************/
hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgeqrf_bufferSize(handle->handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgeqrf_bufferSize(handle->handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnCgeqrf_bufferSize(handle->handle, m, n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnZgeqrf_bufferSize(handle->handle, m, n, (cuDoubleComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            tau,
                                  float*            work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSgeqrf(handle->handle, m, n, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           tau,
                                  double*           work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDgeqrf(handle->handle, m, n, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  tau,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgeqrf(handle->handle,
                                            m,
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)tau,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* tau,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgeqrf(handle->handle,
                                            m,
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESV ********************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              float*            A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              float*            B,
                                                              int               ldb,
                                                              float*            X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSSgesv_bufferSize(
        handle->handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, nullptr, lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              double*           A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              double*           B,
                                                              int               ldb,
                                                              double*           X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDDgesv_bufferSize(
        handle->handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, nullptr, lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              hipFloatComplex*  A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              hipFloatComplex*  B,
                                                              int               ldb,
                                                              hipFloatComplex*  X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCCgesv_bufferSize(handle->handle,
                                                       n,
                                                       nrhs,
                                                       (cuComplex*)A,
                                                       lda,
                                                       devIpiv,
                                                       (cuComplex*)B,
                                                       ldb,
                                                       (cuComplex*)X,
                                                       ldx,
                                                       nullptr,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgesv_bufferSize(hipsolverHandle_t handle,
                                                              int               n,
                                                              int               nrhs,
                                                              hipDoubleComplex* A,
                                                              int               lda,
                                                              int*              devIpiv,
                                                              hipDoubleComplex* B,
                                                              int               ldb,
                                                              hipDoubleComplex* X,
                                                              int               ldx,
                                                              size_t*           lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZZgesv_bufferSize(handle->handle,
                                                       n,
                                                       nrhs,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       devIpiv,
                                                       (cuDoubleComplex*)B,
                                                       ldb,
                                                       (cuDoubleComplex*)X,
                                                       ldx,
                                                       nullptr,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSSgesv(
        handle->handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDDgesv(
        handle->handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   hipFloatComplex*  B,
                                                   int               ldb,
                                                   hipFloatComplex*  X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCCgesv(handle->handle,
                                            n,
                                            nrhs,
                                            (cuComplex*)A,
                                            lda,
                                            devIpiv,
                                            (cuComplex*)B,
                                            ldb,
                                            (cuComplex*)X,
                                            ldx,
                                            work,
                                            lwork,
                                            niters,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgesv(hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   int*              devIpiv,
                                                   hipDoubleComplex* B,
                                                   int               ldb,
                                                   hipDoubleComplex* X,
                                                   int               ldx,
                                                   void*             work,
                                                   size_t            lwork,
                                                   int*              niters,
                                                   int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZZgesv(handle->handle,
                                            n,
                                            nrhs,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            devIpiv,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            (cuDoubleComplex*)X,
                                            ldx,
                                            work,
                                            lwork,
                                            niters,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVD ********************/
hipsolverStatus_t hipsolverSgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvd_bufferSize(handle->handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvd(hipsolverHandle_t handle,
                                  signed char       jobu,
                                  signed char       jobv,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            S,
                                  float*            U,
                                  int               ldu,
                                  float*            V,
                                  int               ldv,
                                  float*            work,
                                  int               lwork,
                                  float*            rwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvd(
        handle->handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvd(hipsolverHandle_t handle,
                                  signed char       jobu,
                                  signed char       jobv,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           S,
                                  double*           U,
                                  int               ldu,
                                  double*           V,
                                  int               ldv,
                                  double*           work,
                                  int               lwork,
                                  double*           rwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgesvd(
        handle->handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvd(hipsolverHandle_t handle,
                                  signed char       jobu,
                                  signed char       jobv,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  float*            S,
                                  hipFloatComplex*  U,
                                  int               ldu,
                                  hipFloatComplex*  V,
                                  int               ldv,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  float*            rwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgesvd(handle->handle,
                                            jobu,
                                            jobv,
                                            m,
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            S,
                                            (cuComplex*)U,
                                            ldu,
                                            (cuComplex*)V,
                                            ldv,
                                            (cuComplex*)work,
                                            lwork,
                                            rwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvd(hipsolverHandle_t handle,
                                  signed char       jobu,
                                  signed char       jobv,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  double*           S,
                                  hipDoubleComplex* U,
                                  int               ldu,
                                  hipDoubleComplex* V,
                                  int               ldv,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  double*           rwork,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgesvd(handle->handle,
                                            jobu,
                                            jobv,
                                            m,
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            S,
                                            (cuDoubleComplex*)U,
                                            ldu,
                                            (cuDoubleComplex*)V,
                                            ldv,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            rwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVDJ ********************/
hipsolverStatus_t hipsolverSgesvdj_bufferSize(hipsolverDnHandle_t   handle,
                                              hipsolverEigMode_t    jobz,
                                              int                   econ,
                                              int                   m,
                                              int                   n,
                                              const float*          A,
                                              int                   lda,
                                              const float*          S,
                                              const float*          U,
                                              int                   ldu,
                                              const float*          V,
                                              int                   ldv,
                                              int*                  lwork,
                                              hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSgesvdj_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        econ,
                                                        m,
                                                        n,
                                                        A,
                                                        lda,
                                                        S,
                                                        U,
                                                        ldu,
                                                        V,
                                                        ldv,
                                                        lwork,
                                                        info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdj_bufferSize(hipsolverDnHandle_t   handle,
                                              hipsolverEigMode_t    jobz,
                                              int                   econ,
                                              int                   m,
                                              int                   n,
                                              const double*         A,
                                              int                   lda,
                                              const double*         S,
                                              const double*         U,
                                              int                   ldu,
                                              const double*         V,
                                              int                   ldv,
                                              int*                  lwork,
                                              hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDgesvdj_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        econ,
                                                        m,
                                                        n,
                                                        A,
                                                        lda,
                                                        S,
                                                        U,
                                                        ldu,
                                                        V,
                                                        ldv,
                                                        lwork,
                                                        info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdj_bufferSize(hipsolverDnHandle_t    handle,
                                              hipsolverEigMode_t     jobz,
                                              int                    econ,
                                              int                    m,
                                              int                    n,
                                              const hipFloatComplex* A,
                                              int                    lda,
                                              const float*           S,
                                              const hipFloatComplex* U,
                                              int                    ldu,
                                              const hipFloatComplex* V,
                                              int                    ldv,
                                              int*                   lwork,
                                              hipsolverGesvdjInfo_t  info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCgesvdj_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        econ,
                                                        m,
                                                        n,
                                                        (cuComplex*)A,
                                                        lda,
                                                        S,
                                                        (cuComplex*)U,
                                                        ldu,
                                                        (cuComplex*)V,
                                                        ldv,
                                                        lwork,
                                                        info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdj_bufferSize(hipsolverDnHandle_t     handle,
                                              hipsolverEigMode_t      jobz,
                                              int                     econ,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              const double*           S,
                                              const hipDoubleComplex* U,
                                              int                     ldu,
                                              const hipDoubleComplex* V,
                                              int                     ldv,
                                              int*                    lwork,
                                              hipsolverGesvdjInfo_t   info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZgesvdj_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        econ,
                                                        m,
                                                        n,
                                                        (cuDoubleComplex*)A,
                                                        lda,
                                                        S,
                                                        (cuDoubleComplex*)U,
                                                        ldu,
                                                        (cuDoubleComplex*)V,
                                                        ldv,
                                                        lwork,
                                                        info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvdj(hipsolverDnHandle_t   handle,
                                   hipsolverEigMode_t    jobz,
                                   int                   econ,
                                   int                   m,
                                   int                   n,
                                   float*                A,
                                   int                   lda,
                                   float*                S,
                                   float*                U,
                                   int                   ldu,
                                   float*                V,
                                   int                   ldv,
                                   float*                work,
                                   int                   lwork,
                                   int*                  devInfo,
                                   hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSgesvdj(handle->handle,
                                             hip2cuda_evect(jobz),
                                             econ,
                                             m,
                                             n,
                                             A,
                                             lda,
                                             S,
                                             U,
                                             ldu,
                                             V,
                                             ldv,
                                             work,
                                             lwork,
                                             devInfo,
                                             info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdj(hipsolverDnHandle_t   handle,
                                   hipsolverEigMode_t    jobz,
                                   int                   econ,
                                   int                   m,
                                   int                   n,
                                   double*               A,
                                   int                   lda,
                                   double*               S,
                                   double*               U,
                                   int                   ldu,
                                   double*               V,
                                   int                   ldv,
                                   double*               work,
                                   int                   lwork,
                                   int*                  devInfo,
                                   hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDgesvdj(handle->handle,
                                             hip2cuda_evect(jobz),
                                             econ,
                                             m,
                                             n,
                                             A,
                                             lda,
                                             S,
                                             U,
                                             ldu,
                                             V,
                                             ldv,
                                             work,
                                             lwork,
                                             devInfo,
                                             info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdj(hipsolverDnHandle_t   handle,
                                   hipsolverEigMode_t    jobz,
                                   int                   econ,
                                   int                   m,
                                   int                   n,
                                   hipFloatComplex*      A,
                                   int                   lda,
                                   float*                S,
                                   hipFloatComplex*      U,
                                   int                   ldu,
                                   hipFloatComplex*      V,
                                   int                   ldv,
                                   hipFloatComplex*      work,
                                   int                   lwork,
                                   int*                  devInfo,
                                   hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCgesvdj(handle->handle,
                                             hip2cuda_evect(jobz),
                                             econ,
                                             m,
                                             n,
                                             (cuComplex*)A,
                                             lda,
                                             S,
                                             (cuComplex*)U,
                                             ldu,
                                             (cuComplex*)V,
                                             ldv,
                                             (cuComplex*)work,
                                             lwork,
                                             devInfo,
                                             info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdj(hipsolverDnHandle_t   handle,
                                   hipsolverEigMode_t    jobz,
                                   int                   econ,
                                   int                   m,
                                   int                   n,
                                   hipDoubleComplex*     A,
                                   int                   lda,
                                   double*               S,
                                   hipDoubleComplex*     U,
                                   int                   ldu,
                                   hipDoubleComplex*     V,
                                   int                   ldv,
                                   hipDoubleComplex*     work,
                                   int                   lwork,
                                   int*                  devInfo,
                                   hipsolverGesvdjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZgesvdj(handle->handle,
                                             hip2cuda_evect(jobz),
                                             econ,
                                             m,
                                             n,
                                             (cuDoubleComplex*)A,
                                             lda,
                                             S,
                                             (cuDoubleComplex*)U,
                                             ldu,
                                             (cuDoubleComplex*)V,
                                             ldv,
                                             (cuDoubleComplex*)work,
                                             lwork,
                                             devInfo,
                                             info->info));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVDJ_BATCHED ********************/
hipsolverStatus_t hipsolverSgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
                                                     hipsolverEigMode_t    jobz,
                                                     int                   m,
                                                     int                   n,
                                                     const float*          A,
                                                     int                   lda,
                                                     const float*          S,
                                                     const float*          U,
                                                     int                   ldu,
                                                     const float*          V,
                                                     int                   ldv,
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t info,
                                                     int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSgesvdjBatched_bufferSize(handle->handle,
                                                               hip2cuda_evect(jobz),
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               S,
                                                               U,
                                                               ldu,
                                                               V,
                                                               ldv,
                                                               lwork,
                                                               info->info,
                                                               batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
                                                     hipsolverEigMode_t    jobz,
                                                     int                   m,
                                                     int                   n,
                                                     const double*         A,
                                                     int                   lda,
                                                     const double*         S,
                                                     const double*         U,
                                                     int                   ldu,
                                                     const double*         V,
                                                     int                   ldv,
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t info,
                                                     int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDgesvdjBatched_bufferSize(handle->handle,
                                                               hip2cuda_evect(jobz),
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               S,
                                                               U,
                                                               ldu,
                                                               V,
                                                               ldv,
                                                               lwork,
                                                               info->info,
                                                               batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdjBatched_bufferSize(hipsolverDnHandle_t    handle,
                                                     hipsolverEigMode_t     jobz,
                                                     int                    m,
                                                     int                    n,
                                                     const hipFloatComplex* A,
                                                     int                    lda,
                                                     const float*           S,
                                                     const hipFloatComplex* U,
                                                     int                    ldu,
                                                     const hipFloatComplex* V,
                                                     int                    ldv,
                                                     int*                   lwork,
                                                     hipsolverGesvdjInfo_t  info,
                                                     int                    batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCgesvdjBatched_bufferSize(handle->handle,
                                                               hip2cuda_evect(jobz),
                                                               m,
                                                               n,
                                                               (cuComplex*)A,
                                                               lda,
                                                               S,
                                                               (cuComplex*)U,
                                                               ldu,
                                                               (cuComplex*)V,
                                                               ldv,
                                                               lwork,
                                                               info->info,
                                                               batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdjBatched_bufferSize(hipsolverDnHandle_t     handle,
                                                     hipsolverEigMode_t      jobz,
                                                     int                     m,
                                                     int                     n,
                                                     const hipDoubleComplex* A,
                                                     int                     lda,
                                                     const double*           S,
                                                     const hipDoubleComplex* U,
                                                     int                     ldu,
                                                     const hipDoubleComplex* V,
                                                     int                     ldv,
                                                     int*                    lwork,
                                                     hipsolverGesvdjInfo_t   info,
                                                     int                     batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZgesvdjBatched_bufferSize(handle->handle,
                                                               hip2cuda_evect(jobz),
                                                               m,
                                                               n,
                                                               (cuDoubleComplex*)A,
                                                               lda,
                                                               S,
                                                               (cuDoubleComplex*)U,
                                                               ldu,
                                                               (cuDoubleComplex*)V,
                                                               ldv,
                                                               lwork,
                                                               info->info,
                                                               batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgesvdjBatched(hipsolverDnHandle_t   handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   m,
                                          int                   n,
                                          float*                A,
                                          int                   lda,
                                          float*                S,
                                          float*                U,
                                          int                   ldu,
                                          float*                V,
                                          int                   ldv,
                                          float*                work,
                                          int                   lwork,
                                          int*                  devInfo,
                                          hipsolverGesvdjInfo_t info,
                                          int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSgesvdjBatched(handle->handle,
                                                    hip2cuda_evect(jobz),
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    S,
                                                    U,
                                                    ldu,
                                                    V,
                                                    ldv,
                                                    work,
                                                    lwork,
                                                    devInfo,
                                                    info->info,
                                                    batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgesvdjBatched(hipsolverDnHandle_t   handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   m,
                                          int                   n,
                                          double*               A,
                                          int                   lda,
                                          double*               S,
                                          double*               U,
                                          int                   ldu,
                                          double*               V,
                                          int                   ldv,
                                          double*               work,
                                          int                   lwork,
                                          int*                  devInfo,
                                          hipsolverGesvdjInfo_t info,
                                          int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDgesvdjBatched(handle->handle,
                                                    hip2cuda_evect(jobz),
                                                    m,
                                                    n,
                                                    A,
                                                    lda,
                                                    S,
                                                    U,
                                                    ldu,
                                                    V,
                                                    ldv,
                                                    work,
                                                    lwork,
                                                    devInfo,
                                                    info->info,
                                                    batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgesvdjBatched(hipsolverDnHandle_t   handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   m,
                                          int                   n,
                                          hipFloatComplex*      A,
                                          int                   lda,
                                          float*                S,
                                          hipFloatComplex*      U,
                                          int                   ldu,
                                          hipFloatComplex*      V,
                                          int                   ldv,
                                          hipFloatComplex*      work,
                                          int                   lwork,
                                          int*                  devInfo,
                                          hipsolverGesvdjInfo_t info,
                                          int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCgesvdjBatched(handle->handle,
                                                    hip2cuda_evect(jobz),
                                                    m,
                                                    n,
                                                    (cuComplex*)A,
                                                    lda,
                                                    S,
                                                    (cuComplex*)U,
                                                    ldu,
                                                    (cuComplex*)V,
                                                    ldv,
                                                    (cuComplex*)work,
                                                    lwork,
                                                    devInfo,
                                                    info->info,
                                                    batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgesvdjBatched(hipsolverDnHandle_t   handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   m,
                                          int                   n,
                                          hipDoubleComplex*     A,
                                          int                   lda,
                                          double*               S,
                                          hipDoubleComplex*     U,
                                          int                   ldu,
                                          hipDoubleComplex*     V,
                                          int                   ldv,
                                          hipDoubleComplex*     work,
                                          int                   lwork,
                                          int*                  devInfo,
                                          hipsolverGesvdjInfo_t info,
                                          int                   batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZgesvdjBatched(handle->handle,
                                                    hip2cuda_evect(jobz),
                                                    m,
                                                    n,
                                                    (cuDoubleComplex*)A,
                                                    lda,
                                                    S,
                                                    (cuDoubleComplex*)U,
                                                    ldu,
                                                    (cuDoubleComplex*)V,
                                                    ldv,
                                                    (cuDoubleComplex*)work,
                                                    lwork,
                                                    devInfo,
                                                    info->info,
                                                    batch_count));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVDA_STRIDED_BATCHED ********************/
hipsolverStatus_t hipsolverDnSgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                              hipsolverEigMode_t jobz,
                                                              int                rank,
                                                              int                m,
                                                              int                n,
                                                              const float*       A,
                                                              int                lda,
                                                              long long int      strideA,
                                                              const float*       S,
                                                              long long int      strideS,
                                                              const float*       U,
                                                              int                ldu,
                                                              long long int      strideU,
                                                              const float*       V,
                                                              int                ldv,
                                                              long long int      strideV,
                                                              int*               lwork,
                                                              int                batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvdaStridedBatched_bufferSize(handle->handle,
                                                                      hip2cuda_evect(jobz),
                                                                      rank,
                                                                      m,
                                                                      n,
                                                                      A,
                                                                      lda,
                                                                      strideA,
                                                                      S,
                                                                      strideS,
                                                                      U,
                                                                      ldu,
                                                                      strideU,
                                                                      V,
                                                                      ldv,
                                                                      strideV,
                                                                      lwork,
                                                                      batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                              hipsolverEigMode_t jobz,
                                                              int                rank,
                                                              int                m,
                                                              int                n,
                                                              const double*      A,
                                                              int                lda,
                                                              long long int      strideA,
                                                              const double*      S,
                                                              long long int      strideS,
                                                              const double*      U,
                                                              int                ldu,
                                                              long long int      strideU,
                                                              const double*      V,
                                                              int                ldv,
                                                              long long int      strideV,
                                                              int*               lwork,
                                                              int                batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgesvdaStridedBatched_bufferSize(handle->handle,
                                                                      hip2cuda_evect(jobz),
                                                                      rank,
                                                                      m,
                                                                      n,
                                                                      A,
                                                                      lda,
                                                                      strideA,
                                                                      S,
                                                                      strideS,
                                                                      U,
                                                                      ldu,
                                                                      strideU,
                                                                      V,
                                                                      ldv,
                                                                      strideV,
                                                                      lwork,
                                                                      batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnCgesvdaStridedBatched_bufferSize(hipsolverHandle_t      handle,
                                                              hipsolverEigMode_t     jobz,
                                                              int                    rank,
                                                              int                    m,
                                                              int                    n,
                                                              const hipFloatComplex* A,
                                                              int                    lda,
                                                              long long int          strideA,
                                                              const float*           S,
                                                              long long int          strideS,
                                                              const hipFloatComplex* U,
                                                              int                    ldu,
                                                              long long int          strideU,
                                                              const hipFloatComplex* V,
                                                              int                    ldv,
                                                              long long int          strideV,
                                                              int*                   lwork,
                                                              int                    batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgesvdaStridedBatched_bufferSize(handle->handle,
                                                                      hip2cuda_evect(jobz),
                                                                      rank,
                                                                      m,
                                                                      n,
                                                                      (cuComplex*)A,
                                                                      lda,
                                                                      strideA,
                                                                      S,
                                                                      strideS,
                                                                      (cuComplex*)U,
                                                                      ldu,
                                                                      strideU,
                                                                      (cuComplex*)V,
                                                                      ldv,
                                                                      strideV,
                                                                      lwork,
                                                                      batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZgesvdaStridedBatched_bufferSize(hipsolverHandle_t       handle,
                                                              hipsolverEigMode_t      jobz,
                                                              int                     rank,
                                                              int                     m,
                                                              int                     n,
                                                              const hipDoubleComplex* A,
                                                              int                     lda,
                                                              long long int           strideA,
                                                              const double*           S,
                                                              long long int           strideS,
                                                              const hipDoubleComplex* U,
                                                              int                     ldu,
                                                              long long int           strideU,
                                                              const hipDoubleComplex* V,
                                                              int                     ldv,
                                                              long long int           strideV,
                                                              int*                    lwork,
                                                              int                     batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgesvdaStridedBatched_bufferSize(handle->handle,
                                                                      hip2cuda_evect(jobz),
                                                                      rank,
                                                                      m,
                                                                      n,
                                                                      (cuDoubleComplex*)A,
                                                                      lda,
                                                                      strideA,
                                                                      S,
                                                                      strideS,
                                                                      (cuDoubleComplex*)U,
                                                                      ldu,
                                                                      strideU,
                                                                      (cuDoubleComplex*)V,
                                                                      ldv,
                                                                      strideV,
                                                                      lwork,
                                                                      batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnSgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                   hipsolverEigMode_t jobz,
                                                   int                rank,
                                                   int                m,
                                                   int                n,
                                                   const float*       A,
                                                   int                lda,
                                                   long long int      strideA,
                                                   float*             S,
                                                   long long int      strideS,
                                                   float*             U,
                                                   int                ldu,
                                                   long long int      strideU,
                                                   float*             V,
                                                   int                ldv,
                                                   long long int      strideV,
                                                   float*             work,
                                                   int                lwork,
                                                   int*               devInfo,
                                                   double*            hRnrmF,
                                                   int                batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgesvdaStridedBatched(handle->handle,
                                                           hip2cuda_evect(jobz),
                                                           rank,
                                                           m,
                                                           n,
                                                           A,
                                                           lda,
                                                           strideA,
                                                           S,
                                                           strideS,
                                                           U,
                                                           ldu,
                                                           strideU,
                                                           V,
                                                           ldv,
                                                           strideV,
                                                           work,
                                                           lwork,
                                                           devInfo,
                                                           hRnrmF,
                                                           batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                   hipsolverEigMode_t jobz,
                                                   int                rank,
                                                   int                m,
                                                   int                n,
                                                   const double*      A,
                                                   int                lda,
                                                   long long int      strideA,
                                                   double*            S,
                                                   long long int      strideS,
                                                   double*            U,
                                                   int                ldu,
                                                   long long int      strideU,
                                                   double*            V,
                                                   int                ldv,
                                                   long long int      strideV,
                                                   double*            work,
                                                   int                lwork,
                                                   int*               devInfo,
                                                   double*            hRnrmF,
                                                   int                batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgesvdaStridedBatched(handle->handle,
                                                           hip2cuda_evect(jobz),
                                                           rank,
                                                           m,
                                                           n,
                                                           A,
                                                           lda,
                                                           strideA,
                                                           S,
                                                           strideS,
                                                           U,
                                                           ldu,
                                                           strideU,
                                                           V,
                                                           ldv,
                                                           strideV,
                                                           work,
                                                           lwork,
                                                           devInfo,
                                                           hRnrmF,
                                                           batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnCgesvdaStridedBatched(hipsolverHandle_t      handle,
                                                   hipsolverEigMode_t     jobz,
                                                   int                    rank,
                                                   int                    m,
                                                   int                    n,
                                                   const hipFloatComplex* A,
                                                   int                    lda,
                                                   long long int          strideA,
                                                   float*                 S,
                                                   long long int          strideS,
                                                   hipFloatComplex*       U,
                                                   int                    ldu,
                                                   long long int          strideU,
                                                   hipFloatComplex*       V,
                                                   int                    ldv,
                                                   long long int          strideV,
                                                   hipFloatComplex*       work,
                                                   int                    lwork,
                                                   int*                   devInfo,
                                                   double*                hRnrmF,
                                                   int                    batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgesvdaStridedBatched(handle->handle,
                                                           hip2cuda_evect(jobz),
                                                           rank,
                                                           m,
                                                           n,
                                                           (cuComplex*)A,
                                                           lda,
                                                           strideA,
                                                           S,
                                                           strideS,
                                                           (cuComplex*)U,
                                                           ldu,
                                                           strideU,
                                                           (cuComplex*)V,
                                                           ldv,
                                                           strideV,
                                                           (cuComplex*)work,
                                                           lwork,
                                                           devInfo,
                                                           hRnrmF,
                                                           batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZgesvdaStridedBatched(hipsolverHandle_t       handle,
                                                   hipsolverEigMode_t      jobz,
                                                   int                     rank,
                                                   int                     m,
                                                   int                     n,
                                                   const hipDoubleComplex* A,
                                                   int                     lda,
                                                   long long int           strideA,
                                                   double*                 S,
                                                   long long int           strideS,
                                                   hipDoubleComplex*       U,
                                                   int                     ldu,
                                                   long long int           strideU,
                                                   hipDoubleComplex*       V,
                                                   int                     ldv,
                                                   long long int           strideV,
                                                   hipDoubleComplex*       work,
                                                   int                     lwork,
                                                   int*                    devInfo,
                                                   double*                 hRnrmF,
                                                   int                     batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgesvdaStridedBatched(handle->handle,
                                                           hip2cuda_evect(jobz),
                                                           rank,
                                                           m,
                                                           n,
                                                           (cuDoubleComplex*)A,
                                                           lda,
                                                           strideA,
                                                           S,
                                                           strideS,
                                                           (cuDoubleComplex*)U,
                                                           ldu,
                                                           strideU,
                                                           (cuDoubleComplex*)V,
                                                           ldv,
                                                           strideV,
                                                           (cuDoubleComplex*)work,
                                                           lwork,
                                                           devInfo,
                                                           hRnrmF,
                                                           batch_count));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GETRF ********************/
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgetrf_bufferSize(handle->handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgetrf_bufferSize(handle->handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnCgetrf_bufferSize(handle->handle, m, n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnZgetrf_bufferSize(handle->handle, m, n, (cuDoubleComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgetrf(handle->handle, m, n, A, lda, work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgetrf(handle->handle, m, n, A, lda, work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipFloatComplex*  A,
                                  int               lda,
                                  hipFloatComplex*  work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgetrf(
        handle->handle, m, n, (cuComplex*)A, lda, (cuComplex*)work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipDoubleComplex* A,
                                  int               lda,
                                  hipDoubleComplex* work,
                                  int               lwork,
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgetrf(
        handle->handle, m, n, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverSgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             float*               A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             float*               B,
                                             int                  ldb,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             double*              A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             double*              B,
                                             int                  ldb,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             hipFloatComplex*     B,
                                             int                  ldb,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrs_bufferSize(hipsolverHandle_t    handle,
                                             hipsolverOperation_t trans,
                                             int                  n,
                                             int                  nrhs,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             int*                 devIpiv,
                                             hipDoubleComplex*    B,
                                             int                  ldb,
                                             int*                 lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  float*               A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  float*               B,
                                  int                  ldb,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSgetrs(
        handle->handle, hip2cuda_operation(trans), n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  double*              A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  double*              B,
                                  int                  ldb,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDgetrs(
        handle->handle, hip2cuda_operation(trans), n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipFloatComplex*     B,
                                  int                  ldb,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCgetrs(handle->handle,
                                            hip2cuda_operation(trans),
                                            n,
                                            nrhs,
                                            (cuComplex*)A,
                                            lda,
                                            devIpiv,
                                            (cuComplex*)B,
                                            ldb,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipDoubleComplex*    B,
                                  int                  ldb,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZgetrs(handle->handle,
                                            hip2cuda_operation(trans),
                                            n,
                                            nrhs,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            devIpiv,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** POTRF ********************/
hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSpotrf_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDpotrf_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotrf_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotrf_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuDoubleComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSpotrf(handle->handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDpotrf(handle->handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotrf(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotrf(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** POTRF_BATCHED ********************/
hipsolverStatus_t hipsolverSpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipFloatComplex*    A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipDoubleComplex*   A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         float*              work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSpotrfBatched(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A[],
                                         int                 lda,
                                         double*             work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDpotrfBatched(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipFloatComplex*    A[],
                                         int                 lda,
                                         hipFloatComplex*    work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotrfBatched(
        handle->handle, hip2cuda_fill(uplo), n, (cuComplex**)A, lda, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipDoubleComplex*   A[],
                                         int                 lda,
                                         hipDoubleComplex*   work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotrfBatched(
        handle->handle, hip2cuda_fill(uplo), n, (cuDoubleComplex**)A, lda, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

/******************** POTRI ********************/
hipsolverStatus_t hipsolverSpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSpotri_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDpotri_bufferSize(handle->handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotri_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotri_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotri_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotri_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuDoubleComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSpotri(handle->handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDpotri(handle->handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotri(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotri(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotri(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** POTRS ********************/
hipsolverStatus_t hipsolverSpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             float*              A,
                                             int                 lda,
                                             float*              B,
                                             int                 ldb,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             double*             A,
                                             int                 lda,
                                             double*             B,
                                             int                 ldb,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             hipFloatComplex*    B,
                                             int                 ldb,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrs_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             hipDoubleComplex*   B,
                                             int                 ldb,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  float*              A,
                                  int                 lda,
                                  float*              B,
                                  int                 ldb,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnSpotrs(handle->handle, hip2cuda_fill(uplo), n, nrhs, A, lda, B, ldb, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  double*             A,
                                  int                 lda,
                                  double*             B,
                                  int                 ldb,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnDpotrs(handle->handle, hip2cuda_fill(uplo), n, nrhs, A, lda, B, ldb, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  hipFloatComplex*    B,
                                  int                 ldb,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotrs(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            nrhs,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)B,
                                            ldb,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrs(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  int                 nrhs,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  hipDoubleComplex*   B,
                                  int                 ldb,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotrs(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            nrhs,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** POTRS_BATCHED ********************/
hipsolverStatus_t hipsolverSpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A[],
                                                    int                 lda,
                                                    float*              B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A[],
                                                    int                 lda,
                                                    double*             B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipFloatComplex*    A[],
                                                    int                 lda,
                                                    hipFloatComplex*    B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipDoubleComplex*   A[],
                                                    int                 lda,
                                                    hipDoubleComplex*   B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpotrsBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A[],
                                         int                 lda,
                                         float*              B[],
                                         int                 ldb,
                                         float*              work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSpotrsBatched(
        handle->handle, hip2cuda_fill(uplo), n, nrhs, A, lda, B, ldb, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrsBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A[],
                                         int                 lda,
                                         double*             B[],
                                         int                 ldb,
                                         double*             work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDpotrsBatched(
        handle->handle, hip2cuda_fill(uplo), n, nrhs, A, lda, B, ldb, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrsBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipFloatComplex*    A[],
                                         int                 lda,
                                         hipFloatComplex*    B[],
                                         int                 ldb,
                                         hipFloatComplex*    work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCpotrsBatched(handle->handle,
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   nrhs,
                                                   (cuComplex**)A,
                                                   lda,
                                                   (cuComplex**)B,
                                                   ldb,
                                                   devInfo,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrsBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipDoubleComplex*   A[],
                                         int                 lda,
                                         hipDoubleComplex*   B[],
                                         int                 ldb,
                                         hipDoubleComplex*   work,
                                         int                 lwork,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZpotrsBatched(handle->handle,
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   nrhs,
                                                   (cuDoubleComplex**)A,
                                                   lda,
                                                   (cuDoubleComplex**)B,
                                                   ldb,
                                                   devInfo,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYEVD/HEEVD ********************/
hipsolverStatus_t hipsolverSsyevd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverEigMode_t  jobz,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A,
                                             int                 lda,
                                             float*              W,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsyevd_bufferSize(
        handle->handle, hip2cuda_evect(jobz), hip2cuda_fill(uplo), n, A, lda, W, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverEigMode_t  jobz,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A,
                                             int                 lda,
                                             double*             W,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsyevd_bufferSize(
        handle->handle, hip2cuda_evect(jobz), hip2cuda_fill(uplo), n, A, lda, W, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverEigMode_t  jobz,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             float*              W,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCheevd_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverEigMode_t  jobz,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             double*             W,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZheevd_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              W,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsyevd(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            W,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             W,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsyevd(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            W,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              W,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCheevd(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            W,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t   handle,
                                  hipsolverEigMode_t  jobz,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             W,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZheevd(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            W,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYEVDX/HEEVDX ********************/
hipsolverStatus_t hipsolverDnSsyevdx_bufferSize(hipsolverHandle_t   handle,
                                                hipsolverEigMode_t  jobz,
                                                hipsolverEigRange_t range,
                                                hipsolverFillMode_t uplo,
                                                int                 n,
                                                const float*        A,
                                                int                 lda,
                                                float               vl,
                                                float               vu,
                                                int                 il,
                                                int                 iu,
                                                int*                nev,
                                                const float*        W,
                                                int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsyevdx_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDsyevdx_bufferSize(hipsolverHandle_t   handle,
                                                hipsolverEigMode_t  jobz,
                                                hipsolverEigRange_t range,
                                                hipsolverFillMode_t uplo,
                                                int                 n,
                                                const double*       A,
                                                int                 lda,
                                                double              vl,
                                                double              vu,
                                                int                 il,
                                                int                 iu,
                                                int*                nev,
                                                const double*       W,
                                                int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsyevdx_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnCheevdx_bufferSize(hipsolverHandle_t      handle,
                                                hipsolverEigMode_t     jobz,
                                                hipsolverEigRange_t    range,
                                                hipsolverFillMode_t    uplo,
                                                int                    n,
                                                const hipFloatComplex* A,
                                                int                    lda,
                                                float                  vl,
                                                float                  vu,
                                                int                    il,
                                                int                    iu,
                                                int*                   nev,
                                                const float*           W,
                                                int*                   lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCheevdx_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        (cuComplex*)A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZheevdx_bufferSize(hipsolverHandle_t       handle,
                                                hipsolverEigMode_t      jobz,
                                                hipsolverEigRange_t     range,
                                                hipsolverFillMode_t     uplo,
                                                int                     n,
                                                const hipDoubleComplex* A,
                                                int                     lda,
                                                double                  vl,
                                                double                  vu,
                                                int                     il,
                                                int                     iu,
                                                int*                    nev,
                                                const double*           W,
                                                int*                    lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZheevdx_bufferSize(handle->handle,
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        (cuDoubleComplex*)A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnSsyevdx(hipsolverHandle_t   handle,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     float*              A,
                                     int                 lda,
                                     float               vl,
                                     float               vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     float*              W,
                                     float*              work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsyevdx(handle->handle,
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             A,
                                             lda,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDsyevdx(hipsolverHandle_t   handle,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     double*             W,
                                     double*             work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsyevdx(handle->handle,
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             A,
                                             lda,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnCheevdx(hipsolverHandle_t   handle,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     hipFloatComplex*    A,
                                     int                 lda,
                                     float               vl,
                                     float               vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     float*              W,
                                     hipFloatComplex*    work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCheevdx(handle->handle,
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             (cuComplex*)A,
                                             lda,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             (cuComplex*)work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZheevdx(hipsolverHandle_t   handle,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     hipDoubleComplex*   A,
                                     int                 lda,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     double*             W,
                                     hipDoubleComplex*   work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZheevdx(handle->handle,
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             (cuDoubleComplex*)A,
                                             lda,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             (cuDoubleComplex*)work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYEVJ/HEEVJ ********************/
hipsolverStatus_t hipsolverSsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             float*               A,
                                             int                  lda,
                                             float*               W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsyevj_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             double*              A,
                                             int                  lda,
                                             double*              W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsyevj_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevj_bufferSize(hipsolverDnHandle_t  handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             float*               W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCheevj_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevj_bufferSize(hipsolverDnHandle_t  handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             double*              W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZheevj_bufferSize(handle->handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevj(hipsolverDnHandle_t  handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  float*               A,
                                  int                  lda,
                                  float*               W,
                                  float*               work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsyevj(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            W,
                                            work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevj(hipsolverDnHandle_t  handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  double*              A,
                                  int                  lda,
                                  double*              W,
                                  double*              work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsyevj(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            W,
                                            work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevj(hipsolverDnHandle_t  handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipFloatComplex*     A,
                                  int                  lda,
                                  float*               W,
                                  hipFloatComplex*     work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCheevj(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            W,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevj(hipsolverDnHandle_t  handle,
                                  hipsolverEigMode_t   jobz,
                                  hipsolverFillMode_t  uplo,
                                  int                  n,
                                  hipDoubleComplex*    A,
                                  int                  lda,
                                  double*              W,
                                  hipDoubleComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo,
                                  hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZheevj(handle->handle,
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            W,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYEVJ_BATCHED/HEEVJ_BATCHED ********************/
hipsolverStatus_t hipsolverSsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    float*               A,
                                                    int                  lda,
                                                    float*               W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t info,
                                                    int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsyevjBatched_bufferSize(handle->handle,
                                                              hip2cuda_evect(jobz),
                                                              hip2cuda_fill(uplo),
                                                              n,
                                                              A,
                                                              lda,
                                                              W,
                                                              lwork,
                                                              info->info,
                                                              batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    double*              A,
                                                    int                  lda,
                                                    double*              W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t info,
                                                    int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsyevjBatched_bufferSize(handle->handle,
                                                              hip2cuda_evect(jobz),
                                                              hip2cuda_fill(uplo),
                                                              n,
                                                              A,
                                                              lda,
                                                              W,
                                                              lwork,
                                                              info->info,
                                                              batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    hipFloatComplex*     A,
                                                    int                  lda,
                                                    float*               W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t info,
                                                    int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCheevjBatched_bufferSize(handle->handle,
                                                              hip2cuda_evect(jobz),
                                                              hip2cuda_fill(uplo),
                                                              n,
                                                              (cuComplex*)A,
                                                              lda,
                                                              W,
                                                              lwork,
                                                              info->info,
                                                              batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                    hipsolverEigMode_t   jobz,
                                                    hipsolverFillMode_t  uplo,
                                                    int                  n,
                                                    hipDoubleComplex*    A,
                                                    int                  lda,
                                                    double*              W,
                                                    int*                 lwork,
                                                    hipsolverSyevjInfo_t info,
                                                    int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZheevjBatched_bufferSize(handle->handle,
                                                              hip2cuda_evect(jobz),
                                                              hip2cuda_fill(uplo),
                                                              n,
                                                              (cuDoubleComplex*)A,
                                                              lda,
                                                              W,
                                                              lwork,
                                                              info->info,
                                                              batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSsyevjBatched(hipsolverDnHandle_t  handle,
                                         hipsolverEigMode_t   jobz,
                                         hipsolverFillMode_t  uplo,
                                         int                  n,
                                         float*               A,
                                         int                  lda,
                                         float*               W,
                                         float*               work,
                                         int                  lwork,
                                         int*                 devInfo,
                                         hipsolverSyevjInfo_t info,
                                         int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsyevjBatched(handle->handle,
                                                   hip2cuda_evect(jobz),
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   A,
                                                   lda,
                                                   W,
                                                   work,
                                                   lwork,
                                                   devInfo,
                                                   info->info,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsyevjBatched(hipsolverDnHandle_t  handle,
                                         hipsolverEigMode_t   jobz,
                                         hipsolverFillMode_t  uplo,
                                         int                  n,
                                         double*              A,
                                         int                  lda,
                                         double*              W,
                                         double*              work,
                                         int                  lwork,
                                         int*                 devInfo,
                                         hipsolverSyevjInfo_t info,
                                         int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsyevjBatched(handle->handle,
                                                   hip2cuda_evect(jobz),
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   A,
                                                   lda,
                                                   W,
                                                   work,
                                                   lwork,
                                                   devInfo,
                                                   info->info,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevjBatched(hipsolverDnHandle_t  handle,
                                         hipsolverEigMode_t   jobz,
                                         hipsolverFillMode_t  uplo,
                                         int                  n,
                                         hipFloatComplex*     A,
                                         int                  lda,
                                         float*               W,
                                         hipFloatComplex*     work,
                                         int                  lwork,
                                         int*                 devInfo,
                                         hipsolverSyevjInfo_t info,
                                         int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnCheevjBatched(handle->handle,
                                                   hip2cuda_evect(jobz),
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   (cuComplex*)A,
                                                   lda,
                                                   W,
                                                   (cuComplex*)work,
                                                   lwork,
                                                   devInfo,
                                                   info->info,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZheevjBatched(hipsolverDnHandle_t  handle,
                                         hipsolverEigMode_t   jobz,
                                         hipsolverFillMode_t  uplo,
                                         int                  n,
                                         hipDoubleComplex*    A,
                                         int                  lda,
                                         double*              W,
                                         hipDoubleComplex*    work,
                                         int                  lwork,
                                         int*                 devInfo,
                                         hipsolverSyevjInfo_t info,
                                         int                  batch_count)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZheevjBatched(handle->handle,
                                                   hip2cuda_evect(jobz),
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   (cuDoubleComplex*)A,
                                                   lda,
                                                   W,
                                                   (cuDoubleComplex*)work,
                                                   lwork,
                                                   devInfo,
                                                   info->info,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYGVD/HEGVD ********************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigType_t  itype,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              B,
                                                              int                 ldb,
                                                              float*              W,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsygvd_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigType_t  itype,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             B,
                                                              int                 ldb,
                                                              double*             W,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsygvd_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigType_t  itype,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    B,
                                                              int                 ldb,
                                                              float*              W,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChegvd_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       (cuComplex*)B,
                                                       ldb,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigType_t  itype,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   B,
                                                              int                 ldb,
                                                              double*             W,
                                                              int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhegvd_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)B,
                                                       ldb,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd(hipsolverHandle_t   handle,
                                                   hipsolverEigType_t  itype,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              B,
                                                   int                 ldb,
                                                   float*              W,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsygvd(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            W,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd(hipsolverHandle_t   handle,
                                                   hipsolverEigType_t  itype,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             B,
                                                   int                 ldb,
                                                   double*             W,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsygvd(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            W,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd(hipsolverHandle_t   handle,
                                                   hipsolverEigType_t  itype,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    B,
                                                   int                 ldb,
                                                   float*              W,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChegvd(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)B,
                                            ldb,
                                            W,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd(hipsolverHandle_t   handle,
                                                   hipsolverEigType_t  itype,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   B,
                                                   int                 ldb,
                                                   double*             W,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhegvd(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            W,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYGVDX/HEGVDX ********************/
hipsolverStatus_t hipsolverDnSsygvdx_bufferSize(hipsolverHandle_t   handle,
                                                hipsolverEigType_t  itype,
                                                hipsolverEigMode_t  jobz,
                                                hipsolverEigRange_t range,
                                                hipsolverFillMode_t uplo,
                                                int                 n,
                                                const float*        A,
                                                int                 lda,
                                                const float*        B,
                                                int                 ldb,
                                                float               vl,
                                                float               vu,
                                                int                 il,
                                                int                 iu,
                                                int*                nev,
                                                const float*        W,
                                                int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsygvdx_bufferSize(handle->handle,
                                                        hip2cuda_eform(itype),
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDsygvdx_bufferSize(hipsolverHandle_t   handle,
                                                hipsolverEigType_t  itype,
                                                hipsolverEigMode_t  jobz,
                                                hipsolverEigRange_t range,
                                                hipsolverFillMode_t uplo,
                                                int                 n,
                                                const double*       A,
                                                int                 lda,
                                                const double*       B,
                                                int                 ldb,
                                                double              vl,
                                                double              vu,
                                                int                 il,
                                                int                 iu,
                                                int*                nev,
                                                const double*       W,
                                                int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsygvdx_bufferSize(handle->handle,
                                                        hip2cuda_eform(itype),
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnChegvdx_bufferSize(hipsolverHandle_t      handle,
                                                hipsolverEigType_t     itype,
                                                hipsolverEigMode_t     jobz,
                                                hipsolverEigRange_t    range,
                                                hipsolverFillMode_t    uplo,
                                                int                    n,
                                                const hipFloatComplex* A,
                                                int                    lda,
                                                const hipFloatComplex* B,
                                                int                    ldb,
                                                float                  vl,
                                                float                  vu,
                                                int                    il,
                                                int                    iu,
                                                int*                   nev,
                                                const float*           W,
                                                int*                   lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChegvdx_bufferSize(handle->handle,
                                                        hip2cuda_eform(itype),
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        (cuComplex*)A,
                                                        lda,
                                                        (cuComplex*)B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZhegvdx_bufferSize(hipsolverHandle_t       handle,
                                                hipsolverEigType_t      itype,
                                                hipsolverEigMode_t      jobz,
                                                hipsolverEigRange_t     range,
                                                hipsolverFillMode_t     uplo,
                                                int                     n,
                                                const hipDoubleComplex* A,
                                                int                     lda,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                double                  vl,
                                                double                  vu,
                                                int                     il,
                                                int                     iu,
                                                int*                    nev,
                                                const double*           W,
                                                int*                    lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhegvdx_bufferSize(handle->handle,
                                                        hip2cuda_eform(itype),
                                                        hip2cuda_evect(jobz),
                                                        hip2cuda_erange(range),
                                                        hip2cuda_fill(uplo),
                                                        n,
                                                        (cuDoubleComplex*)A,
                                                        lda,
                                                        (cuDoubleComplex*)B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        nev,
                                                        W,
                                                        lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnSsygvdx(hipsolverHandle_t   handle,
                                     hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     float*              A,
                                     int                 lda,
                                     float*              B,
                                     int                 ldb,
                                     float               vl,
                                     float               vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     float*              W,
                                     float*              work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsygvdx(handle->handle,
                                             hip2cuda_eform(itype),
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             A,
                                             lda,
                                             B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDsygvdx(hipsolverHandle_t   handle,
                                     hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     double*             A,
                                     int                 lda,
                                     double*             B,
                                     int                 ldb,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     double*             W,
                                     double*             work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsygvdx(handle->handle,
                                             hip2cuda_eform(itype),
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             A,
                                             lda,
                                             B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnChegvdx(hipsolverHandle_t   handle,
                                     hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     hipFloatComplex*    A,
                                     int                 lda,
                                     hipFloatComplex*    B,
                                     int                 ldb,
                                     float               vl,
                                     float               vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     float*              W,
                                     hipFloatComplex*    work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChegvdx(handle->handle,
                                             hip2cuda_eform(itype),
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             (cuComplex*)A,
                                             lda,
                                             (cuComplex*)B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             (cuComplex*)work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnZhegvdx(hipsolverHandle_t   handle,
                                     hipsolverEigType_t  itype,
                                     hipsolverEigMode_t  jobz,
                                     hipsolverEigRange_t range,
                                     hipsolverFillMode_t uplo,
                                     int                 n,
                                     hipDoubleComplex*   A,
                                     int                 lda,
                                     hipDoubleComplex*   B,
                                     int                 ldb,
                                     double              vl,
                                     double              vu,
                                     int                 il,
                                     int                 iu,
                                     int*                nev,
                                     double*             W,
                                     hipDoubleComplex*   work,
                                     int                 lwork,
                                     int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhegvdx(handle->handle,
                                             hip2cuda_eform(itype),
                                             hip2cuda_evect(jobz),
                                             hip2cuda_erange(range),
                                             hip2cuda_fill(uplo),
                                             n,
                                             (cuDoubleComplex*)A,
                                             lda,
                                             (cuDoubleComplex*)B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             (cuDoubleComplex*)work,
                                             lwork,
                                             devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYGVJ/HEGVJ ********************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvj_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverEigType_t   itype,
                                                              hipsolverEigMode_t   jobz,
                                                              hipsolverFillMode_t  uplo,
                                                              int                  n,
                                                              float*               A,
                                                              int                  lda,
                                                              float*               B,
                                                              int                  ldb,
                                                              float*               W,
                                                              int*                 lwork,
                                                              hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsygvj_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvj_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverEigType_t   itype,
                                                              hipsolverEigMode_t   jobz,
                                                              hipsolverFillMode_t  uplo,
                                                              int                  n,
                                                              double*              A,
                                                              int                  lda,
                                                              double*              B,
                                                              int                  ldb,
                                                              double*              W,
                                                              int*                 lwork,
                                                              hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsygvj_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       B,
                                                       ldb,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvj_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverEigType_t   itype,
                                                              hipsolverEigMode_t   jobz,
                                                              hipsolverFillMode_t  uplo,
                                                              int                  n,
                                                              hipFloatComplex*     A,
                                                              int                  lda,
                                                              hipFloatComplex*     B,
                                                              int                  ldb,
                                                              float*               W,
                                                              int*                 lwork,
                                                              hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnChegvj_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       (cuComplex*)B,
                                                       ldb,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvj_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverEigType_t   itype,
                                                              hipsolverEigMode_t   jobz,
                                                              hipsolverFillMode_t  uplo,
                                                              int                  n,
                                                              hipDoubleComplex*    A,
                                                              int                  lda,
                                                              hipDoubleComplex*    B,
                                                              int                  ldb,
                                                              double*              W,
                                                              int*                 lwork,
                                                              hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZhegvj_bufferSize(handle->handle,
                                                       hip2cuda_eform(itype),
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       (cuDoubleComplex*)B,
                                                       ldb,
                                                       W,
                                                       lwork,
                                                       info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvj(hipsolverHandle_t    handle,
                                                   hipsolverEigType_t   itype,
                                                   hipsolverEigMode_t   jobz,
                                                   hipsolverFillMode_t  uplo,
                                                   int                  n,
                                                   float*               A,
                                                   int                  lda,
                                                   float*               B,
                                                   int                  ldb,
                                                   float*               W,
                                                   float*               work,
                                                   int                  lwork,
                                                   int*                 devInfo,
                                                   hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnSsygvj(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            W,
                                            work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvj(hipsolverHandle_t    handle,
                                                   hipsolverEigType_t   itype,
                                                   hipsolverEigMode_t   jobz,
                                                   hipsolverFillMode_t  uplo,
                                                   int                  n,
                                                   double*              A,
                                                   int                  lda,
                                                   double*              B,
                                                   int                  ldb,
                                                   double*              W,
                                                   double*              work,
                                                   int                  lwork,
                                                   int*                 devInfo,
                                                   hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnDsygvj(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            B,
                                            ldb,
                                            W,
                                            work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvj(hipsolverHandle_t    handle,
                                                   hipsolverEigType_t   itype,
                                                   hipsolverEigMode_t   jobz,
                                                   hipsolverFillMode_t  uplo,
                                                   int                  n,
                                                   hipFloatComplex*     A,
                                                   int                  lda,
                                                   hipFloatComplex*     B,
                                                   int                  ldb,
                                                   float*               W,
                                                   hipFloatComplex*     work,
                                                   int                  lwork,
                                                   int*                 devInfo,
                                                   hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnChegvj(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            (cuComplex*)B,
                                            ldb,
                                            W,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvj(hipsolverHandle_t    handle,
                                                   hipsolverEigType_t   itype,
                                                   hipsolverEigMode_t   jobz,
                                                   hipsolverFillMode_t  uplo,
                                                   int                  n,
                                                   hipDoubleComplex*    A,
                                                   int                  lda,
                                                   hipDoubleComplex*    B,
                                                   int                  ldb,
                                                   double*              W,
                                                   hipDoubleComplex*    work,
                                                   int                  lwork,
                                                   int*                 devInfo,
                                                   hipsolverSyevjInfo_t info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return cuda2hip_status(cusolverDnZhegvj(handle->handle,
                                            hip2cuda_eform(itype),
                                            hip2cuda_evect(jobz),
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)B,
                                            ldb,
                                            W,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo,
                                            info->info));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYTRD/HETRD ********************/
hipsolverStatus_t hipsolverSsytrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A,
                                             int                 lda,
                                             float*              D,
                                             float*              E,
                                             float*              tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsytrd_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A,
                                             int                 lda,
                                             double*             D,
                                             double*             E,
                                             double*             tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsytrd_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A,
                                             int                 lda,
                                             float*              D,
                                             float*              E,
                                             hipFloatComplex*    tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChetrd_bufferSize(
        handle->handle, hip2cuda_fill(uplo), n, (cuComplex*)A, lda, D, E, (cuComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A,
                                             int                 lda,
                                             double*             D,
                                             double*             E,
                                             hipDoubleComplex*   tau,
                                             int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhetrd_bufferSize(handle->handle,
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuDoubleComplex*)A,
                                                       lda,
                                                       D,
                                                       E,
                                                       (cuDoubleComplex*)tau,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  float*              tau,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsytrd(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  double*             tau,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsytrd(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  hipFloatComplex*    tau,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnChetrd(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            D,
                                            E,
                                            (cuComplex*)tau,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  double*             D,
                                  double*             E,
                                  hipDoubleComplex*   tau,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZhetrd(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            D,
                                            E,
                                            (cuDoubleComplex*)tau,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** SYTRF ********************/
hipsolverStatus_t
    hipsolverSsytrf_bufferSize(hipsolverHandle_t handle, int n, float* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsytrf_bufferSize(handle->handle, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t
    hipsolverDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsytrf_bufferSize(handle->handle, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnCsytrf_bufferSize(handle->handle, n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(
        cusolverDnZsytrf_bufferSize(handle->handle, n, (cuDoubleComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  float*              A,
                                  int                 lda,
                                  int*                ipiv,
                                  float*              work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnSsytrf(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, ipiv, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  double*             A,
                                  int                 lda,
                                  int*                ipiv,
                                  double*             work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnDsytrf(
        handle->handle, hip2cuda_fill(uplo), n, A, lda, ipiv, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipFloatComplex*    A,
                                  int                 lda,
                                  int*                ipiv,
                                  hipFloatComplex*    work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnCsytrf(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuComplex*)A,
                                            lda,
                                            ipiv,
                                            (cuComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZsytrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipDoubleComplex*   A,
                                  int                 lda,
                                  int*                ipiv,
                                  hipDoubleComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return cuda2hip_status(cusolverDnZsytrf(handle->handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            ipiv,
                                            (cuDoubleComplex*)work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

} // extern C
