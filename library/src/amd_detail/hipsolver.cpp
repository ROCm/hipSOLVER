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
 *  \brief Implementation of the hipSOLVER regular APIs on the rocSOLVER side.
 */

#include "hipsolver.h"
#include "error_macros.hpp"
#include "exceptions.hpp"
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

using namespace std;

extern "C" {

// The following functions are not included in the public API of rocSOLVER and must be declared

rocblas_status rocsolver_sgesv_outofplace(rocblas_handle    handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float*            A,
                                          const rocblas_int lda,
                                          rocblas_int*      ipiv,
                                          float*            B,
                                          const rocblas_int ldb,
                                          float*            X,
                                          const rocblas_int ldx,
                                          rocblas_int*      info);

rocblas_status rocsolver_dgesv_outofplace(rocblas_handle    handle,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double*           A,
                                          const rocblas_int lda,
                                          rocblas_int*      ipiv,
                                          double*           B,
                                          const rocblas_int ldb,
                                          double*           X,
                                          const rocblas_int ldx,
                                          rocblas_int*      info);

rocblas_status rocsolver_cgesv_outofplace(rocblas_handle         handle,
                                          const rocblas_int      n,
                                          const rocblas_int      nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int      lda,
                                          rocblas_int*           ipiv,
                                          rocblas_float_complex* B,
                                          const rocblas_int      ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int      ldx,
                                          rocblas_int*           info);

rocblas_status rocsolver_zgesv_outofplace(rocblas_handle          handle,
                                          const rocblas_int       n,
                                          const rocblas_int       nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int       lda,
                                          rocblas_int*            ipiv,
                                          rocblas_double_complex* B,
                                          const rocblas_int       ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int       ldx,
                                          rocblas_int*            info);

rocblas_status rocsolver_sgels_outofplace(rocblas_handle    handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          float*            A,
                                          const rocblas_int lda,
                                          float*            B,
                                          const rocblas_int ldb,
                                          float*            X,
                                          const rocblas_int ldx,
                                          rocblas_int*      info);

rocblas_status rocsolver_dgels_outofplace(rocblas_handle    handle,
                                          rocblas_operation trans,
                                          const rocblas_int m,
                                          const rocblas_int n,
                                          const rocblas_int nrhs,
                                          double*           A,
                                          const rocblas_int lda,
                                          double*           B,
                                          const rocblas_int ldb,
                                          double*           X,
                                          const rocblas_int ldx,
                                          rocblas_int*      info);

rocblas_status rocsolver_cgels_outofplace(rocblas_handle         handle,
                                          rocblas_operation      trans,
                                          const rocblas_int      m,
                                          const rocblas_int      n,
                                          const rocblas_int      nrhs,
                                          rocblas_float_complex* A,
                                          const rocblas_int      lda,
                                          rocblas_float_complex* B,
                                          const rocblas_int      ldb,
                                          rocblas_float_complex* X,
                                          const rocblas_int      ldx,
                                          rocblas_int*           info);

rocblas_status rocsolver_zgels_outofplace(rocblas_handle          handle,
                                          rocblas_operation       trans,
                                          const rocblas_int       m,
                                          const rocblas_int       n,
                                          const rocblas_int       nrhs,
                                          rocblas_double_complex* A,
                                          const rocblas_int       lda,
                                          rocblas_double_complex* B,
                                          const rocblas_int       ldb,
                                          rocblas_double_complex* X,
                                          const rocblas_int       ldx,
                                          rocblas_int*            info);

rocblas_status rocsolver_ssyevdx_inplace(rocblas_handle       handle,
                                         const rocblas_evect  evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill   uplo,
                                         const rocblas_int    n,
                                         float*               A,
                                         const rocblas_int    lda,
                                         const float          vl,
                                         const float          vu,
                                         const rocblas_int    il,
                                         const rocblas_int    iu,
                                         const float          abstol,
                                         rocblas_int*         nev,
                                         float*               W,
                                         rocblas_int*         info);

rocblas_status rocsolver_dsyevdx_inplace(rocblas_handle       handle,
                                         const rocblas_evect  evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill   uplo,
                                         const rocblas_int    n,
                                         double*              A,
                                         const rocblas_int    lda,
                                         const double         vl,
                                         const double         vu,
                                         const rocblas_int    il,
                                         const rocblas_int    iu,
                                         const double         abstol,
                                         rocblas_int*         nev,
                                         double*              W,
                                         rocblas_int*         info);

rocblas_status rocsolver_cheevdx_inplace(rocblas_handle         handle,
                                         const rocblas_evect    evect,
                                         const rocblas_erange   erange,
                                         const rocblas_fill     uplo,
                                         const rocblas_int      n,
                                         rocblas_float_complex* A,
                                         const rocblas_int      lda,
                                         const float            vl,
                                         const float            vu,
                                         const rocblas_int      il,
                                         const rocblas_int      iu,
                                         const float            abstol,
                                         rocblas_int*           nev,
                                         float*                 W,
                                         rocblas_int*           info);

rocblas_status rocsolver_zheevdx_inplace(rocblas_handle          handle,
                                         const rocblas_evect     evect,
                                         const rocblas_erange    erange,
                                         const rocblas_fill      uplo,
                                         const rocblas_int       n,
                                         rocblas_double_complex* A,
                                         const rocblas_int       lda,
                                         const double            vl,
                                         const double            vu,
                                         const rocblas_int       il,
                                         const rocblas_int       iu,
                                         const double            abstol,
                                         rocblas_int*            nev,
                                         double*                 W,
                                         rocblas_int*            info);

rocblas_status rocsolver_ssygvdx_inplace(rocblas_handle       handle,
                                         const rocblas_eform  itype,
                                         const rocblas_evect  evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill   uplo,
                                         const rocblas_int    n,
                                         float*               A,
                                         const rocblas_int    lda,
                                         float*               B,
                                         const rocblas_int    ldb,
                                         const float          vl,
                                         const float          vu,
                                         const rocblas_int    il,
                                         const rocblas_int    iu,
                                         const float          abstol,
                                         rocblas_int*         h_nev,
                                         float*               W,
                                         rocblas_int*         info);

rocblas_status rocsolver_dsygvdx_inplace(rocblas_handle       handle,
                                         const rocblas_eform  itype,
                                         const rocblas_evect  evect,
                                         const rocblas_erange erange,
                                         const rocblas_fill   uplo,
                                         const rocblas_int    n,
                                         double*              A,
                                         const rocblas_int    lda,
                                         double*              B,
                                         const rocblas_int    ldb,
                                         const double         vl,
                                         const double         vu,
                                         const rocblas_int    il,
                                         const rocblas_int    iu,
                                         const double         abstol,
                                         rocblas_int*         h_nev,
                                         double*              W,
                                         rocblas_int*         info);

rocblas_status rocsolver_chegvdx_inplace(rocblas_handle         handle,
                                         const rocblas_eform    itype,
                                         const rocblas_evect    evect,
                                         const rocblas_erange   erange,
                                         const rocblas_fill     uplo,
                                         const rocblas_int      n,
                                         rocblas_float_complex* A,
                                         const rocblas_int      lda,
                                         rocblas_float_complex* B,
                                         const rocblas_int      ldb,
                                         const float            vl,
                                         const float            vu,
                                         const rocblas_int      il,
                                         const rocblas_int      iu,
                                         const float            abstol,
                                         rocblas_int*           h_nev,
                                         float*                 W,
                                         rocblas_int*           info);

rocblas_status rocsolver_zhegvdx_inplace(rocblas_handle          handle,
                                         const rocblas_eform     itype,
                                         const rocblas_evect     evect,
                                         const rocblas_erange    erange,
                                         const rocblas_fill      uplo,
                                         const rocblas_int       n,
                                         rocblas_double_complex* A,
                                         const rocblas_int       lda,
                                         rocblas_double_complex* B,
                                         const rocblas_int       ldb,
                                         const double            vl,
                                         const double            vu,
                                         const rocblas_int       il,
                                         const rocblas_int       iu,
                                         const double            abstol,
                                         rocblas_int*            h_nev,
                                         double*                 W,
                                         rocblas_int*            info);

rocblas_status rocsolver_sgesvdj_notransv(rocblas_handle      handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int   m,
                                          const rocblas_int   n,
                                          float*              A,
                                          const rocblas_int   lda,
                                          const float         abstol,
                                          float*              residual,
                                          const rocblas_int   max_sweeps,
                                          rocblas_int*        n_sweeps,
                                          float*              S,
                                          float*              U,
                                          const rocblas_int   ldu,
                                          float*              V,
                                          const rocblas_int   ldv,
                                          rocblas_int*        info);

rocblas_status rocsolver_dgesvdj_notransv(rocblas_handle      handle,
                                          const rocblas_svect left_svect,
                                          const rocblas_svect right_svect,
                                          const rocblas_int   m,
                                          const rocblas_int   n,
                                          double*             A,
                                          const rocblas_int   lda,
                                          const double        abstol,
                                          double*             residual,
                                          const rocblas_int   max_sweeps,
                                          rocblas_int*        n_sweeps,
                                          double*             S,
                                          double*             U,
                                          const rocblas_int   ldu,
                                          double*             V,
                                          const rocblas_int   ldv,
                                          rocblas_int*        info);

rocblas_status rocsolver_cgesvdj_notransv(rocblas_handle         handle,
                                          const rocblas_svect    left_svect,
                                          const rocblas_svect    right_svect,
                                          const rocblas_int      m,
                                          const rocblas_int      n,
                                          rocblas_float_complex* A,
                                          const rocblas_int      lda,
                                          const float            abstol,
                                          float*                 residual,
                                          const rocblas_int      max_sweeps,
                                          rocblas_int*           n_sweeps,
                                          float*                 S,
                                          rocblas_float_complex* U,
                                          const rocblas_int      ldu,
                                          rocblas_float_complex* V,
                                          const rocblas_int      ldv,
                                          rocblas_int*           info);

rocblas_status rocsolver_zgesvdj_notransv(rocblas_handle          handle,
                                          const rocblas_svect     left_svect,
                                          const rocblas_svect     right_svect,
                                          const rocblas_int       m,
                                          const rocblas_int       n,
                                          rocblas_double_complex* A,
                                          const rocblas_int       lda,
                                          const double            abstol,
                                          double*                 residual,
                                          const rocblas_int       max_sweeps,
                                          rocblas_int*            n_sweeps,
                                          double*                 S,
                                          rocblas_double_complex* U,
                                          const rocblas_int       ldu,
                                          rocblas_double_complex* V,
                                          const rocblas_int       ldv,
                                          rocblas_int*            info);

rocblas_status rocsolver_sgesvdj_notransv_strided_batched(rocblas_handle       handle,
                                                          const rocblas_svect  left_svect,
                                                          const rocblas_svect  right_svect,
                                                          const rocblas_int    m,
                                                          const rocblas_int    n,
                                                          float*               A,
                                                          const rocblas_int    lda,
                                                          const rocblas_stride strideA,
                                                          const float          abstol,
                                                          float*               residual,
                                                          const rocblas_int    max_sweeps,
                                                          rocblas_int*         n_sweeps,
                                                          float*               S,
                                                          const rocblas_stride strideS,
                                                          float*               U,
                                                          const rocblas_int    ldu,
                                                          const rocblas_stride strideU,
                                                          float*               V,
                                                          const rocblas_int    ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int*         info,
                                                          const rocblas_int    batch_count);

rocblas_status rocsolver_dgesvdj_notransv_strided_batched(rocblas_handle       handle,
                                                          const rocblas_svect  left_svect,
                                                          const rocblas_svect  right_svect,
                                                          const rocblas_int    m,
                                                          const rocblas_int    n,
                                                          double*              A,
                                                          const rocblas_int    lda,
                                                          const rocblas_stride strideA,
                                                          const double         abstol,
                                                          double*              residual,
                                                          const rocblas_int    max_sweeps,
                                                          rocblas_int*         n_sweeps,
                                                          double*              S,
                                                          const rocblas_stride strideS,
                                                          double*              U,
                                                          const rocblas_int    ldu,
                                                          const rocblas_stride strideU,
                                                          double*              V,
                                                          const rocblas_int    ldv,
                                                          const rocblas_stride strideV,
                                                          rocblas_int*         info,
                                                          const rocblas_int    batch_count);

rocblas_status rocsolver_cgesvdj_notransv_strided_batched(rocblas_handle         handle,
                                                          const rocblas_svect    left_svect,
                                                          const rocblas_svect    right_svect,
                                                          const rocblas_int      m,
                                                          const rocblas_int      n,
                                                          rocblas_float_complex* A,
                                                          const rocblas_int      lda,
                                                          const rocblas_stride   strideA,
                                                          const float            abstol,
                                                          float*                 residual,
                                                          const rocblas_int      max_sweeps,
                                                          rocblas_int*           n_sweeps,
                                                          float*                 S,
                                                          const rocblas_stride   strideS,
                                                          rocblas_float_complex* U,
                                                          const rocblas_int      ldu,
                                                          const rocblas_stride   strideU,
                                                          rocblas_float_complex* V,
                                                          const rocblas_int      ldv,
                                                          const rocblas_stride   strideV,
                                                          rocblas_int*           info,
                                                          const rocblas_int      batch_count);

rocblas_status rocsolver_zgesvdj_notransv_strided_batched(rocblas_handle          handle,
                                                          const rocblas_svect     left_svect,
                                                          const rocblas_svect     right_svect,
                                                          const rocblas_int       m,
                                                          const rocblas_int       n,
                                                          rocblas_double_complex* A,
                                                          const rocblas_int       lda,
                                                          const rocblas_stride    strideA,
                                                          const double            abstol,
                                                          double*                 residual,
                                                          const rocblas_int       max_sweeps,
                                                          rocblas_int*            n_sweeps,
                                                          double*                 S,
                                                          const rocblas_stride    strideS,
                                                          rocblas_double_complex* U,
                                                          const rocblas_int       ldu,
                                                          const rocblas_stride    strideU,
                                                          rocblas_double_complex* V,
                                                          const rocblas_int       ldv,
                                                          const rocblas_stride    strideV,
                                                          rocblas_int*            info,
                                                          const rocblas_int       batch_count);

/******************** HELPERS ********************/
inline rocblas_status hipsolverManageWorkspace(rocblas_handle handle, size_t new_size)
{
    if(new_size < 0)
        return rocblas_status_memory_error;

    size_t current_size = 0;
    if(rocblas_is_user_managing_device_memory(handle))
        rocblas_get_device_memory_size(handle, &current_size);

    if(new_size > current_size)
        return rocblas_set_device_memory_size(handle, new_size);
    else
        return rocblas_status_success;
}

inline rocblas_status
    hipsolverZeroInfo(rocblas_handle handle, rocblas_int* devInfo, rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!devInfo)
        return rocblas_status_invalid_pointer;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(hipMemsetAsync(devInfo, 0, sizeof(rocblas_int) * batch_count, stream) == hipSuccess)
        return rocblas_status_success;
    else
        return rocblas_status_internal_error;
}

/******************** AUXILIARY ********************/
hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;

    // Create the rocBLAS handle
    return rocblas2hip_status(rocblas_create_handle((rocblas_handle*)handle));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
try
{
    return rocblas2hip_status(rocblas_destroy_handle((rocblas_handle)handle));
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

    return rocblas2hip_status(rocblas_set_stream((rocblas_handle)handle, streamId));
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

    return rocblas2hip_status(rocblas_get_stream((rocblas_handle)handle, streamId));
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

    *info = new hipsolverGesvdjInfo;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(max_sweeps <= 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    info->max_sweeps = max_sweeps;

    return HIPSOLVER_STATUS_SUCCESS;
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

    info->sort_eig = sort_eig;

    return HIPSOLVER_STATUS_SUCCESS;
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

    info->tolerance = tolerance;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!residual)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(info->is_batched)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    if(info->capacity <= 0)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    if(info->is_float)
    {
        float result;
        if(hipMemcpy(&result, info->residual, sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
        *residual = result;
    }
    else
    {
        if(hipMemcpy(residual, info->residual, sizeof(double), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!executed_sweeps)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(info->is_batched)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    if(info->capacity <= 0)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    if(hipMemcpy(executed_sweeps, info->n_sweeps, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return HIPSOLVER_STATUS_SUCCESS;
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

    *info = new hipsolverSyevjInfo;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(max_sweeps <= 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    info->max_sweeps = max_sweeps;

    return HIPSOLVER_STATUS_SUCCESS;
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

    info->sort_eig = sort_eig;

    return HIPSOLVER_STATUS_SUCCESS;
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

    info->tolerance = tolerance;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!residual)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(info->is_batched)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    if(info->capacity <= 0)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    if(info->is_float)
    {
        float result;
        if(hipMemcpy(&result, info->residual, sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
        *residual = result;
    }
    else
    {
        if(hipMemcpy(residual, info->residual, sizeof(double), hipMemcpyDeviceToHost) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!executed_sweeps)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(info->is_batched)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    if(info->capacity <= 0)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    if(hipMemcpy(executed_sweeps, info->n_sweeps, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sorgbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dorgbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cungbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zungbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSorgbr_bufferSize((rocblas_handle)handle, side, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sorgbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDorgbr_bufferSize((rocblas_handle)handle, side, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dorgbr(
        (rocblas_handle)handle, hip2rocblas_side2storev(side), m, n, k, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCungbr_bufferSize((rocblas_handle)handle, side, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cungbr((rocblas_handle)handle,
                                               hip2rocblas_side2storev(side),
                                               m,
                                               n,
                                               k,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZungbr_bufferSize((rocblas_handle)handle, side, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zungbr((rocblas_handle)handle,
                                               hip2rocblas_side2storev(side),
                                               m,
                                               n,
                                               k,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_sorgqr((rocblas_handle)handle, m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dorgqr((rocblas_handle)handle, m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cungqr((rocblas_handle)handle, m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zungqr((rocblas_handle)handle, m, n, k, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSorgqr_bufferSize((rocblas_handle)handle, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sorgqr((rocblas_handle)handle, m, n, k, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDorgqr_bufferSize((rocblas_handle)handle, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dorgqr((rocblas_handle)handle, m, n, k, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCungqr_bufferSize((rocblas_handle)handle, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cungqr((rocblas_handle)handle,
                                               m,
                                               n,
                                               k,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZungqr_bufferSize((rocblas_handle)handle, m, n, k, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zungqr((rocblas_handle)handle,
                                               m,
                                               n,
                                               k,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_sorgtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dorgtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cungtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zungtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSorgtr_bufferSize((rocblas_handle)handle, uplo, n, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_sorgtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDorgtr_bufferSize((rocblas_handle)handle, uplo, n, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_dorgtr((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCungtr_bufferSize((rocblas_handle)handle, uplo, n, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cungtr((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZungtr_bufferSize((rocblas_handle)handle, uplo, n, A, lda, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zungtr((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sormqr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dormqr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cunmqr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zunmqr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSormqr_bufferSize(
            (rocblas_handle)handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sormqr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               k,
                                               A,
                                               lda,
                                               tau,
                                               C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDormqr_bufferSize(
            (rocblas_handle)handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dormqr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               k,
                                               A,
                                               lda,
                                               tau,
                                               C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCunmqr_bufferSize(
            (rocblas_handle)handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cunmqr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               k,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)tau,
                                               (rocblas_float_complex*)C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZunmqr_bufferSize(
            (rocblas_handle)handle, side, trans, m, n, k, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zunmqr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               k,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau,
                                               (rocblas_double_complex*)C,
                                               ldc));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sormtr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_fill(uplo),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dormtr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_fill(uplo),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cunmtr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_fill(uplo),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zunmtr((rocblas_handle)handle,
                                                                   hip2rocblas_side(side),
                                                                   hip2rocblas_fill(uplo),
                                                                   hip2rocblas_operation(trans),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldc));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSormtr_bufferSize(
            (rocblas_handle)handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sormtr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_fill(uplo),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               A,
                                               lda,
                                               tau,
                                               C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDormtr_bufferSize(
            (rocblas_handle)handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dormtr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_fill(uplo),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               A,
                                               lda,
                                               tau,
                                               C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCunmtr_bufferSize(
            (rocblas_handle)handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cunmtr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_fill(uplo),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)tau,
                                               (rocblas_float_complex*)C,
                                               ldc));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZunmtr_bufferSize(
            (rocblas_handle)handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zunmtr((rocblas_handle)handle,
                                               hip2rocblas_side(side),
                                               hip2rocblas_fill(uplo),
                                               hip2rocblas_operation(trans),
                                               m,
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau,
                                               (rocblas_double_complex*)C,
                                               ldc));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sgebrd(
        (rocblas_handle)handle, m, n, nullptr, m, nullptr, nullptr, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dgebrd(
        (rocblas_handle)handle, m, n, nullptr, m, nullptr, nullptr, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cgebrd(
        (rocblas_handle)handle, m, n, nullptr, m, nullptr, nullptr, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zgebrd(
        (rocblas_handle)handle, m, n, nullptr, m, nullptr, nullptr, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSgebrd_bufferSize((rocblas_handle)handle, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_sgebrd((rocblas_handle)handle, m, n, A, lda, D, E, tauq, taup));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDgebrd_bufferSize((rocblas_handle)handle, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_dgebrd((rocblas_handle)handle, m, n, A, lda, D, E, tauq, taup));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCgebrd_bufferSize((rocblas_handle)handle, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cgebrd((rocblas_handle)handle,
                                               m,
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               D,
                                               E,
                                               (rocblas_float_complex*)tauq,
                                               (rocblas_float_complex*)taup));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZgebrd_bufferSize((rocblas_handle)handle, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zgebrd((rocblas_handle)handle,
                                               m,
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               D,
                                               E,
                                               (rocblas_double_complex*)tauq,
                                               (rocblas_double_complex*)taup));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sgels_outofplace((rocblas_handle)handle,
                                                                             rocblas_operation_none,
                                                                             m,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_sgels((rocblas_handle)handle,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       nrhs,
                                       nullptr,
                                       lda,
                                       nullptr,
                                       ldb,
                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dgels_outofplace((rocblas_handle)handle,
                                                                             rocblas_operation_none,
                                                                             m,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_dgels((rocblas_handle)handle,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       nrhs,
                                       nullptr,
                                       lda,
                                       nullptr,
                                       ldb,
                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cgels_outofplace((rocblas_handle)handle,
                                                                             rocblas_operation_none,
                                                                             m,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_cgels((rocblas_handle)handle,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       nrhs,
                                       nullptr,
                                       lda,
                                       nullptr,
                                       ldb,
                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zgels_outofplace((rocblas_handle)handle,
                                                                             rocblas_operation_none,
                                                                             m,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_zgels((rocblas_handle)handle,
                                       rocblas_operation_none,
                                       m,
                                       n,
                                       nrhs,
                                       nullptr,
                                       lda,
                                       nullptr,
                                       ldb,
                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    *lwork = sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSSgels_bufferSize(
            (rocblas_handle)handle, m, n, nrhs, A, lda, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_sgels(
            (rocblas_handle)handle, rocblas_operation_none, m, n, nrhs, A, lda, B, ldb, devInfo));
    else
        return rocblas2hip_status(rocsolver_sgels_outofplace((rocblas_handle)handle,
                                                             rocblas_operation_none,
                                                             m,
                                                             n,
                                                             nrhs,
                                                             A,
                                                             lda,
                                                             B,
                                                             ldb,
                                                             X,
                                                             ldx,
                                                             devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDDgels_bufferSize(
            (rocblas_handle)handle, m, n, nrhs, A, lda, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_dgels(
            (rocblas_handle)handle, rocblas_operation_none, m, n, nrhs, A, lda, B, ldb, devInfo));
    else
        return rocblas2hip_status(rocsolver_dgels_outofplace((rocblas_handle)handle,
                                                             rocblas_operation_none,
                                                             m,
                                                             n,
                                                             nrhs,
                                                             A,
                                                             lda,
                                                             B,
                                                             ldb,
                                                             X,
                                                             ldx,
                                                             devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCCgels_bufferSize(
            (rocblas_handle)handle, m, n, nrhs, A, lda, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_cgels((rocblas_handle)handle,
                                                  rocblas_operation_none,
                                                  m,
                                                  n,
                                                  nrhs,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  devInfo));
    else
        return rocblas2hip_status(rocsolver_cgels_outofplace((rocblas_handle)handle,
                                                             rocblas_operation_none,
                                                             m,
                                                             n,
                                                             nrhs,
                                                             (rocblas_float_complex*)A,
                                                             lda,
                                                             (rocblas_float_complex*)B,
                                                             ldb,
                                                             (rocblas_float_complex*)X,
                                                             ldx,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZZgels_bufferSize(
            (rocblas_handle)handle, m, n, nrhs, A, lda, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_zgels((rocblas_handle)handle,
                                                  rocblas_operation_none,
                                                  m,
                                                  n,
                                                  nrhs,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  devInfo));
    else
        return rocblas2hip_status(rocsolver_zgels_outofplace((rocblas_handle)handle,
                                                             rocblas_operation_none,
                                                             m,
                                                             n,
                                                             nrhs,
                                                             (rocblas_double_complex*)A,
                                                             lda,
                                                             (rocblas_double_complex*)B,
                                                             ldb,
                                                             (rocblas_double_complex*)X,
                                                             ldx,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_sgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_dgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_cgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_zgeqrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSgeqrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sgeqrf((rocblas_handle)handle, m, n, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDgeqrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dgeqrf((rocblas_handle)handle, m, n, A, lda, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCgeqrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cgeqrf(
        (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, (rocblas_float_complex*)tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZgeqrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zgeqrf((rocblas_handle)handle,
                                               m,
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)tau));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sgesv_outofplace((rocblas_handle)handle,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_sgesv(
        (rocblas_handle)handle, n, nrhs, nullptr, lda, nullptr, nullptr, ldb, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dgesv_outofplace((rocblas_handle)handle,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_dgesv(
        (rocblas_handle)handle, n, nrhs, nullptr, lda, nullptr, nullptr, ldb, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cgesv_outofplace((rocblas_handle)handle,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_cgesv(
        (rocblas_handle)handle, n, nrhs, nullptr, lda, nullptr, nullptr, ldb, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    *lwork = sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zgesv_outofplace((rocblas_handle)handle,
                                                                             n,
                                                                             nrhs,
                                                                             nullptr,
                                                                             lda,
                                                                             nullptr,
                                                                             nullptr,
                                                                             ldb,
                                                                             nullptr,
                                                                             ldx,
                                                                             nullptr));
    rocblas2hip_status(rocsolver_zgesv(
        (rocblas_handle)handle, n, nrhs, nullptr, lda, nullptr, nullptr, ldb, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    *lwork = sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSSgesv_bufferSize(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(
            rocsolver_sgesv((rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    else
        return rocblas2hip_status(rocsolver_sgesv_outofplace(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDDgesv_bufferSize(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(
            rocsolver_dgesv((rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, devInfo));
    else
        return rocblas2hip_status(rocsolver_dgesv_outofplace(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCCgesv_bufferSize(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_cgesv((rocblas_handle)handle,
                                                  n,
                                                  nrhs,
                                                  (rocblas_float_complex*)A,
                                                  lda,
                                                  devIpiv,
                                                  (rocblas_float_complex*)B,
                                                  ldb,
                                                  devInfo));
    else
        return rocblas2hip_status(rocsolver_cgesv_outofplace((rocblas_handle)handle,
                                                             n,
                                                             nrhs,
                                                             (rocblas_float_complex*)A,
                                                             lda,
                                                             devIpiv,
                                                             (rocblas_float_complex*)B,
                                                             ldb,
                                                             (rocblas_float_complex*)X,
                                                             ldx,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZZgesv_bufferSize(
            (rocblas_handle)handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(B == X)
        return rocblas2hip_status(rocsolver_zgesv((rocblas_handle)handle,
                                                  n,
                                                  nrhs,
                                                  (rocblas_double_complex*)A,
                                                  lda,
                                                  devIpiv,
                                                  (rocblas_double_complex*)B,
                                                  ldb,
                                                  devInfo));
    else
        return rocblas2hip_status(rocsolver_zgesv_outofplace((rocblas_handle)handle,
                                                             n,
                                                             nrhs,
                                                             (rocblas_double_complex*)A,
                                                             lda,
                                                             devIpiv,
                                                             (rocblas_double_complex*)B,
                                                             ldb,
                                                             (rocblas_double_complex*)X,
                                                             ldx,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sgesvd((rocblas_handle)handle,
                                                                   char2rocblas_svect(jobu),
                                                                   char2rocblas_svect(jobv),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   m,
                                                                   nullptr,
                                                                   nullptr,
                                                                   max(m, 1),
                                                                   nullptr,
                                                                   max(n, 1),
                                                                   nullptr,
                                                                   rocblas_outofplace,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array (aka rwork)
    size_t size_E = min(m, n) > 0 ? sizeof(float) * min(m, n) : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dgesvd((rocblas_handle)handle,
                                                                   char2rocblas_svect(jobu),
                                                                   char2rocblas_svect(jobv),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   m,
                                                                   nullptr,
                                                                   nullptr,
                                                                   max(m, 1),
                                                                   nullptr,
                                                                   max(n, 1),
                                                                   nullptr,
                                                                   rocblas_outofplace,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array (aka rwork)
    size_t size_E = min(m, n) > 0 ? sizeof(double) * min(m, n) : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cgesvd((rocblas_handle)handle,
                                                                   char2rocblas_svect(jobu),
                                                                   char2rocblas_svect(jobv),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   m,
                                                                   nullptr,
                                                                   nullptr,
                                                                   max(m, 1),
                                                                   nullptr,
                                                                   max(n, 1),
                                                                   nullptr,
                                                                   rocblas_outofplace,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array (aka rwork)
    size_t size_E = min(m, n) > 0 ? sizeof(float) * min(m, n) : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zgesvd((rocblas_handle)handle,
                                                                   char2rocblas_svect(jobu),
                                                                   char2rocblas_svect(jobv),
                                                                   m,
                                                                   n,
                                                                   nullptr,
                                                                   m,
                                                                   nullptr,
                                                                   nullptr,
                                                                   max(m, 1),
                                                                   nullptr,
                                                                   max(n, 1),
                                                                   nullptr,
                                                                   rocblas_outofplace,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array (aka rwork)
    size_t size_E = min(m, n) > 0 ? sizeof(double) * min(m, n) : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    rocblas_device_malloc mem((rocblas_handle)handle);

    if(work && lwork)
    {
        if(!rwork && min(m, n) > 1)
        {
            rwork = work;
            work  = rwork + min(m, n);
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSgesvd_bufferSize((rocblas_handle)handle, jobu, jobv, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        if(!rwork && min(m, n) > 1)
        {
            mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * min(m, n));
            if(!mem)
                return HIPSOLVER_STATUS_ALLOC_FAILED;
            rwork = (float*)mem[0];
        }
    }

    return rocblas2hip_status(rocsolver_sgesvd((rocblas_handle)handle,
                                               char2rocblas_svect(jobu),
                                               char2rocblas_svect(jobv),
                                               m,
                                               n,
                                               A,
                                               lda,
                                               S,
                                               U,
                                               ldu,
                                               V,
                                               ldv,
                                               rwork,
                                               rocblas_outofplace,
                                               devInfo));
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
    rocblas_device_malloc mem((rocblas_handle)handle);

    if(work && lwork)
    {
        if(!rwork && min(m, n) > 1)
        {
            rwork = work;
            work  = rwork + min(m, n);
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDgesvd_bufferSize((rocblas_handle)handle, jobu, jobv, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        if(!rwork && min(m, n) > 1)
        {
            mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * min(m, n));
            if(!mem)
                return HIPSOLVER_STATUS_ALLOC_FAILED;
            rwork = (double*)mem[0];
        }
    }

    return rocblas2hip_status(rocsolver_dgesvd((rocblas_handle)handle,
                                               char2rocblas_svect(jobu),
                                               char2rocblas_svect(jobv),
                                               m,
                                               n,
                                               A,
                                               lda,
                                               S,
                                               U,
                                               ldu,
                                               V,
                                               ldv,
                                               rwork,
                                               rocblas_outofplace,
                                               devInfo));
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
    rocblas_device_malloc mem((rocblas_handle)handle);

    if(work && lwork)
    {
        if(!rwork && min(m, n) > 1)
        {
            rwork = (float*)work;
            work  = (hipFloatComplex*)(rwork + min(m, n));
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCgesvd_bufferSize((rocblas_handle)handle, jobu, jobv, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        if(!rwork && min(m, n) > 1)
        {
            mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * min(m, n));
            if(!mem)
                return HIPSOLVER_STATUS_ALLOC_FAILED;
            rwork = (float*)mem[0];
        }
    }

    return rocblas2hip_status(rocsolver_cgesvd((rocblas_handle)handle,
                                               char2rocblas_svect(jobu),
                                               char2rocblas_svect(jobv),
                                               m,
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               S,
                                               (rocblas_float_complex*)U,
                                               ldu,
                                               (rocblas_float_complex*)V,
                                               ldv,
                                               rwork,
                                               rocblas_outofplace,
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
    rocblas_device_malloc mem((rocblas_handle)handle);

    if(work && lwork)
    {
        if(!rwork && min(m, n) > 1)
        {
            rwork = (double*)work;
            work  = (hipDoubleComplex*)(rwork + min(m, n));
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZgesvd_bufferSize((rocblas_handle)handle, jobu, jobv, m, n, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        if(!rwork && min(m, n) > 1)
        {
            mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * min(m, n));
            if(!mem)
                return HIPSOLVER_STATUS_ALLOC_FAILED;
            rwork = (double*)mem[0];
        }
    }

    return rocblas2hip_status(rocsolver_zgesvd((rocblas_handle)handle,
                                               char2rocblas_svect(jobu),
                                               char2rocblas_svect(jobv),
                                               m,
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               S,
                                               (rocblas_double_complex*)U,
                                               ldu,
                                               (rocblas_double_complex*)V,
                                               ldv,
                                               rwork,
                                               rocblas_outofplace,
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_sgesvdj_notransv((rocblas_handle)handle,
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        m,
                                                        n,
                                                        nullptr,
                                                        lda,
                                                        info->tolerance,
                                                        nullptr,
                                                        info->max_sweeps,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        ldu,
                                                        nullptr,
                                                        ldv,
                                                        nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_dgesvdj_notransv((rocblas_handle)handle,
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        m,
                                                        n,
                                                        nullptr,
                                                        lda,
                                                        info->tolerance,
                                                        nullptr,
                                                        info->max_sweeps,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        ldu,
                                                        nullptr,
                                                        ldv,
                                                        nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_cgesvdj_notransv((rocblas_handle)handle,
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        m,
                                                        n,
                                                        nullptr,
                                                        lda,
                                                        info->tolerance,
                                                        nullptr,
                                                        info->max_sweeps,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        ldu,
                                                        nullptr,
                                                        ldv,
                                                        nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_zgesvdj_notransv((rocblas_handle)handle,
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        hip2rocblas_evect2svect(jobz, econ),
                                                        m,
                                                        n,
                                                        nullptr,
                                                        lda,
                                                        info->tolerance,
                                                        nullptr,
                                                        info->max_sweeps,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        ldu,
                                                        nullptr,
                                                        ldv,
                                                        nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSgesvdj_bufferSize(
            (rocblas_handle)handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    // perform computation
    return rocblas2hip_status(rocsolver_sgesvdj_notransv((rocblas_handle)handle,
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         info->tolerance,
                                                         (float*)info->residual,
                                                         info->max_sweeps,
                                                         info->n_sweeps,
                                                         S,
                                                         U,
                                                         ldu,
                                                         V,
                                                         ldv,
                                                         devInfo));
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDgesvdj_bufferSize(
            (rocblas_handle)handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    // perform computation
    return rocblas2hip_status(rocsolver_dgesvdj_notransv((rocblas_handle)handle,
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         info->tolerance,
                                                         info->residual,
                                                         info->max_sweeps,
                                                         info->n_sweeps,
                                                         S,
                                                         U,
                                                         ldu,
                                                         V,
                                                         ldv,
                                                         devInfo));
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCgesvdj_bufferSize(
            (rocblas_handle)handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    // perform computation
    return rocblas2hip_status(rocsolver_cgesvdj_notransv((rocblas_handle)handle,
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         m,
                                                         n,
                                                         (rocblas_float_complex*)A,
                                                         lda,
                                                         info->tolerance,
                                                         (float*)info->residual,
                                                         info->max_sweeps,
                                                         info->n_sweeps,
                                                         S,
                                                         (rocblas_float_complex*)U,
                                                         ldu,
                                                         (rocblas_float_complex*)V,
                                                         ldv,
                                                         devInfo));
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZgesvdj_bufferSize(
            (rocblas_handle)handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    // perform computation
    return rocblas2hip_status(rocsolver_zgesvdj_notransv((rocblas_handle)handle,
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         hip2rocblas_evect2svect(jobz, econ),
                                                         m,
                                                         n,
                                                         (rocblas_double_complex*)A,
                                                         lda,
                                                         info->tolerance,
                                                         info->residual,
                                                         info->max_sweeps,
                                                         info->n_sweeps,
                                                         S,
                                                         (rocblas_double_complex*)U,
                                                         ldu,
                                                         (rocblas_double_complex*)V,
                                                         ldv,
                                                         devInfo));
}
catch(...)
{
    return exception2hip_status();
}

/******************** GESVDJ ********************/
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_sgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   nullptr,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   nullptr,
                                                   info->max_sweeps,
                                                   nullptr,
                                                   nullptr,
                                                   min(m, n),
                                                   nullptr,
                                                   ldu,
                                                   ldu * m,
                                                   nullptr,
                                                   ldv,
                                                   ldv * n,
                                                   nullptr,
                                                   batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   nullptr,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   nullptr,
                                                   info->max_sweeps,
                                                   nullptr,
                                                   nullptr,
                                                   min(m, n),
                                                   nullptr,
                                                   ldu,
                                                   ldu * m,
                                                   nullptr,
                                                   ldv,
                                                   ldv * n,
                                                   nullptr,
                                                   batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   nullptr,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   nullptr,
                                                   info->max_sweeps,
                                                   nullptr,
                                                   nullptr,
                                                   min(m, n),
                                                   nullptr,
                                                   ldu,
                                                   ldu * m,
                                                   nullptr,
                                                   ldv,
                                                   ldv * n,
                                                   nullptr,
                                                   batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   nullptr,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   nullptr,
                                                   info->max_sweeps,
                                                   nullptr,
                                                   nullptr,
                                                   min(m, n),
                                                   nullptr,
                                                   ldu,
                                                   ldu * m,
                                                   nullptr,
                                                   ldv,
                                                   ldv * n,
                                                   nullptr,
                                                   batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSgesvdjBatched_bufferSize((rocblas_handle)handle,
                                                                 jobz,
                                                                 m,
                                                                 n,
                                                                 A,
                                                                 lda,
                                                                 S,
                                                                 U,
                                                                 ldu,
                                                                 V,
                                                                 ldv,
                                                                 &lwork,
                                                                 info,
                                                                 batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = true;

    // perform computation
    return rocblas2hip_status(
        rocsolver_sgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   (float*)info->residual,
                                                   info->max_sweeps,
                                                   info->n_sweeps,
                                                   S,
                                                   min(m, n),
                                                   U,
                                                   ldu,
                                                   ldu * m,
                                                   V,
                                                   ldv,
                                                   ldv * n,
                                                   devInfo,
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDgesvdjBatched_bufferSize((rocblas_handle)handle,
                                                                 jobz,
                                                                 m,
                                                                 n,
                                                                 A,
                                                                 lda,
                                                                 S,
                                                                 U,
                                                                 ldu,
                                                                 V,
                                                                 ldv,
                                                                 &lwork,
                                                                 info,
                                                                 batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = false;

    // perform computation
    return rocblas2hip_status(
        rocsolver_dgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   info->residual,
                                                   info->max_sweeps,
                                                   info->n_sweeps,
                                                   S,
                                                   min(m, n),
                                                   U,
                                                   ldu,
                                                   ldu * m,
                                                   V,
                                                   ldv,
                                                   ldv * n,
                                                   devInfo,
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCgesvdjBatched_bufferSize((rocblas_handle)handle,
                                                                 jobz,
                                                                 m,
                                                                 n,
                                                                 A,
                                                                 lda,
                                                                 S,
                                                                 U,
                                                                 ldu,
                                                                 V,
                                                                 ldv,
                                                                 &lwork,
                                                                 info,
                                                                 batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = true;

    // perform computation
    return rocblas2hip_status(
        rocsolver_cgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   (rocblas_float_complex*)A,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   (float*)info->residual,
                                                   info->max_sweeps,
                                                   info->n_sweeps,
                                                   S,
                                                   min(m, n),
                                                   (rocblas_float_complex*)U,
                                                   ldu,
                                                   ldu * m,
                                                   (rocblas_float_complex*)V,
                                                   ldv,
                                                   ldv * n,
                                                   devInfo,
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

    // prepare workspace
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZgesvdjBatched_bufferSize((rocblas_handle)handle,
                                                                 jobz,
                                                                 m,
                                                                 n,
                                                                 A,
                                                                 lda,
                                                                 S,
                                                                 U,
                                                                 ldu,
                                                                 V,
                                                                 ldv,
                                                                 &lwork,
                                                                 info,
                                                                 batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = false;

    // perform computation
    return rocblas2hip_status(
        rocsolver_zgesvdj_notransv_strided_batched((rocblas_handle)handle,
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   hip2rocblas_evect2svect(jobz, 0),
                                                   m,
                                                   n,
                                                   (rocblas_double_complex*)A,
                                                   lda,
                                                   lda * n,
                                                   info->tolerance,
                                                   info->residual,
                                                   info->max_sweeps,
                                                   info->n_sweeps,
                                                   S,
                                                   min(m, n),
                                                   (rocblas_double_complex*)U,
                                                   ldu,
                                                   ldu * m,
                                                   (rocblas_double_complex*)V,
                                                   ldv,
                                                   ldv * n,
                                                   devInfo,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR && ldv < n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(jobz == HIPSOLVER_EIG_MODE_NOVECTOR && ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    bool use_V_copy = jobz != HIPSOLVER_EIG_MODE_NOVECTOR;
    int  ldv_copy   = use_V_copy ? min(m, n) : 1;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_sgesvdx_strided_batched((rocblas_handle)handle,
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               rocblas_srange_index,
                                                               m,
                                                               n,
                                                               nullptr,
                                                               lda,
                                                               strideA,
                                                               0,
                                                               0,
                                                               1,
                                                               rank,
                                                               nullptr,
                                                               nullptr,
                                                               strideS,
                                                               nullptr,
                                                               ldu,
                                                               strideU,
                                                               nullptr,
                                                               ldv_copy,
                                                               ldv_copy * n,
                                                               nullptr,
                                                               n,
                                                               nullptr,
                                                               batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for nsv array
    size_t size_nsv = sizeof(int) * batch_count;

    // space for V_copy array
    size_t size_V_copy = use_V_copy ? sizeof(float) * ldv_copy * n * batch_count : 0;

    // space for ifail array
    size_t size_ifail = sizeof(int) * min(m, n) * batch_count;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size(
        (rocblas_handle)handle, sz, size_nsv, size_V_copy, size_ifail);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR && ldv < n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(jobz == HIPSOLVER_EIG_MODE_NOVECTOR && ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    bool use_V_copy = jobz != HIPSOLVER_EIG_MODE_NOVECTOR;
    int  ldv_copy   = use_V_copy ? min(m, n) : 1;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_dgesvdx_strided_batched((rocblas_handle)handle,
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               rocblas_srange_index,
                                                               m,
                                                               n,
                                                               nullptr,
                                                               lda,
                                                               strideA,
                                                               0,
                                                               0,
                                                               1,
                                                               rank,
                                                               nullptr,
                                                               nullptr,
                                                               strideS,
                                                               nullptr,
                                                               ldu,
                                                               strideU,
                                                               nullptr,
                                                               ldv_copy,
                                                               ldv_copy * n,
                                                               nullptr,
                                                               n,
                                                               nullptr,
                                                               batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for nsv array
    size_t size_nsv = sizeof(int) * batch_count;

    // space for V_copy array
    size_t size_V_copy = use_V_copy ? sizeof(double) * ldv_copy * n * batch_count : 0;

    // space for ifail array
    size_t size_ifail = sizeof(int) * min(m, n) * batch_count;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size(
        (rocblas_handle)handle, sz, size_nsv, size_V_copy, size_ifail);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR && ldv < n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(jobz == HIPSOLVER_EIG_MODE_NOVECTOR && ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    bool use_V_copy = jobz != HIPSOLVER_EIG_MODE_NOVECTOR;
    int  ldv_copy   = use_V_copy ? min(m, n) : 1;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_cgesvdx_strided_batched((rocblas_handle)handle,
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               rocblas_srange_index,
                                                               m,
                                                               n,
                                                               nullptr,
                                                               lda,
                                                               strideA,
                                                               0,
                                                               0,
                                                               1,
                                                               rank,
                                                               nullptr,
                                                               nullptr,
                                                               strideS,
                                                               nullptr,
                                                               ldu,
                                                               strideU,
                                                               nullptr,
                                                               ldv_copy,
                                                               ldv_copy * n,
                                                               nullptr,
                                                               n,
                                                               nullptr,
                                                               batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for nsv array
    size_t size_nsv = sizeof(int) * batch_count;

    // space for V_copy array
    size_t size_V_copy
        = use_V_copy ? sizeof(rocblas_float_complex) * ldv_copy * n * batch_count : 0;

    // space for ifail array
    size_t size_ifail = sizeof(int) * min(m, n) * batch_count;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size(
        (rocblas_handle)handle, sz, size_nsv, size_V_copy, size_ifail);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR && ldv < n)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(jobz == HIPSOLVER_EIG_MODE_NOVECTOR && ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    bool use_V_copy = jobz != HIPSOLVER_EIG_MODE_NOVECTOR;
    int  ldv_copy   = use_V_copy ? min(m, n) : 1;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_zgesvdx_strided_batched((rocblas_handle)handle,
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               hip2rocblas_evect2svect(jobz, 1),
                                                               rocblas_srange_index,
                                                               m,
                                                               n,
                                                               nullptr,
                                                               lda,
                                                               strideA,
                                                               0,
                                                               0,
                                                               1,
                                                               rank,
                                                               nullptr,
                                                               nullptr,
                                                               strideS,
                                                               nullptr,
                                                               ldu,
                                                               strideU,
                                                               nullptr,
                                                               ldv_copy,
                                                               ldv_copy * n,
                                                               nullptr,
                                                               n,
                                                               nullptr,
                                                               batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for nsv array
    size_t size_nsv = sizeof(int) * batch_count;

    // space for V_copy array
    size_t size_V_copy
        = use_V_copy ? sizeof(rocblas_double_complex) * ldv_copy * n * batch_count : 0;

    // space for ifail array
    size_t size_ifail = sizeof(int) * min(m, n) * batch_count;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size(
        (rocblas_handle)handle, sz, size_nsv, size_V_copy, size_ifail);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocblas_device_malloc mem((rocblas_handle)handle);
    int*                  nsv;
    float*                V_copy;
    int*                  ifail;

    const float one         = 1.0f;
    const float zero        = 0.0f;
    bool        use_V_copy  = false;
    int         ldv_copy    = 1;
    size_t      size_V_copy = 0;
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        if(ldv < n || !V)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        use_V_copy  = true;
        ldv_copy    = min(m, n);
        size_V_copy = sizeof(float) * ldv_copy * n * batch_count;
    }

    // prepare workspace
    if(work && lwork)
    {
        nsv = (int*)work;
        if(batch_count > 0)
            work = (float*)(nsv + batch_count);

        V_copy = work;
        if(use_V_copy)
            work = V_copy + ldv_copy * n * batch_count;

        ifail = (int*)work;
        if(use_V_copy)
            work = (float*)(ifail + n * batch_count);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnSgesvdaStridedBatched_bufferSize((rocblas_handle)handle,
                                                                          jobz,
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
                                                                          &lwork,
                                                                          batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle,
                                    sizeof(int) * batch_count,
                                    size_V_copy,
                                    sizeof(int) * min(m, n) * batch_count);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        nsv    = (int*)mem[0];
        V_copy = (float*)mem[1];
        ifail  = (int*)mem[2];
    }

    // perform computation
    CHECK_ROCBLAS_ERROR(rocsolver_sgesvdx_strided_batched((rocblas_handle)handle,
                                                          hip2rocblas_evect2svect(jobz, 1),
                                                          hip2rocblas_evect2svect(jobz, 1),
                                                          rocblas_srange_index,
                                                          m,
                                                          n,
                                                          const_cast<float*>(A),
                                                          lda,
                                                          strideA,
                                                          0,
                                                          0,
                                                          1,
                                                          rank,
                                                          nsv,
                                                          S,
                                                          strideS,
                                                          U,
                                                          ldu,
                                                          strideU,
                                                          V_copy,
                                                          ldv_copy,
                                                          ldv_copy * n,
                                                          ifail,
                                                          min(m, n),
                                                          devInfo,
                                                          batch_count));

    // transpose V
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
        return rocblas2hip_status(rocblas_sgeam_strided_batched((rocblas_handle)handle,
                                                                rocblas_operation_transpose,
                                                                rocblas_operation_transpose,
                                                                n,
                                                                rank,
                                                                &one,
                                                                V_copy,
                                                                ldv_copy,
                                                                ldv_copy * n,
                                                                &zero,
                                                                V_copy,
                                                                ldv_copy,
                                                                ldv_copy * n,
                                                                V,
                                                                ldv,
                                                                strideV,
                                                                batch_count));
    else
        return HIPSOLVER_STATUS_SUCCESS;
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
    if(ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocblas_device_malloc mem((rocblas_handle)handle);
    int*                  nsv;
    double*               V_copy;
    int*                  ifail;

    const double one         = 1.0f;
    const double zero        = 0.0f;
    bool         use_V_copy  = false;
    int          ldv_copy    = 1;
    size_t       size_V_copy = 0;
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        if(ldv < n || !V)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        use_V_copy  = true;
        ldv_copy    = min(m, n);
        size_V_copy = sizeof(double) * ldv_copy * n * batch_count;
    }

    // prepare workspace
    if(work && lwork)
    {
        nsv = (int*)work;
        if(batch_count > 0)
            work = (double*)(nsv + batch_count);

        V_copy = work;
        if(use_V_copy)
            work = V_copy + ldv_copy * n * batch_count;

        ifail = (int*)work;
        if(use_V_copy)
            work = (double*)(ifail + n * batch_count);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnDgesvdaStridedBatched_bufferSize((rocblas_handle)handle,
                                                                          jobz,
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
                                                                          &lwork,
                                                                          batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle,
                                    sizeof(int) * batch_count,
                                    size_V_copy,
                                    sizeof(int) * min(m, n) * batch_count);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        nsv    = (int*)mem[0];
        V_copy = (double*)mem[1];
        ifail  = (int*)mem[2];
    }

    // perform computation
    CHECK_ROCBLAS_ERROR(rocsolver_dgesvdx_strided_batched((rocblas_handle)handle,
                                                          hip2rocblas_evect2svect(jobz, 1),
                                                          hip2rocblas_evect2svect(jobz, 1),
                                                          rocblas_srange_index,
                                                          m,
                                                          n,
                                                          const_cast<double*>(A),
                                                          lda,
                                                          strideA,
                                                          0,
                                                          0,
                                                          1,
                                                          rank,
                                                          nsv,
                                                          S,
                                                          strideS,
                                                          U,
                                                          ldu,
                                                          strideU,
                                                          V_copy,
                                                          ldv_copy,
                                                          ldv_copy * n,
                                                          ifail,
                                                          min(m, n),
                                                          devInfo,
                                                          batch_count));

    // transpose V
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
        return rocblas2hip_status(rocblas_dgeam_strided_batched((rocblas_handle)handle,
                                                                rocblas_operation_transpose,
                                                                rocblas_operation_transpose,
                                                                n,
                                                                rank,
                                                                &one,
                                                                V_copy,
                                                                ldv_copy,
                                                                ldv_copy * n,
                                                                &zero,
                                                                V_copy,
                                                                ldv_copy,
                                                                ldv_copy * n,
                                                                V,
                                                                ldv,
                                                                strideV,
                                                                batch_count));
    else
        return HIPSOLVER_STATUS_SUCCESS;
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
    if(ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocblas_device_malloc  mem((rocblas_handle)handle);
    int*                   nsv;
    rocblas_float_complex* V_copy;
    int*                   ifail;

    const rocblas_float_complex one         = {1.0f, 0.0f};
    const rocblas_float_complex zero        = {0.0f, 0.0f};
    bool                        use_V_copy  = false;
    int                         ldv_copy    = 1;
    size_t                      size_V_copy = 0;
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        if(ldv < n || !V)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        use_V_copy  = true;
        ldv_copy    = min(m, n);
        size_V_copy = sizeof(rocblas_float_complex) * ldv_copy * n * batch_count;
    }

    // prepare workspace
    if(work && lwork)
    {
        nsv = (int*)work;
        if(batch_count > 0)
            work = (hipFloatComplex*)(nsv + batch_count);

        V_copy = (rocblas_float_complex*)work;
        if(use_V_copy)
            work = (hipFloatComplex*)(V_copy + ldv_copy * n * batch_count);

        ifail = (int*)work;
        if(use_V_copy)
            work = (hipFloatComplex*)(ifail + n * batch_count);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnCgesvdaStridedBatched_bufferSize((rocblas_handle)handle,
                                                                          jobz,
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
                                                                          &lwork,
                                                                          batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle,
                                    sizeof(int) * batch_count,
                                    size_V_copy,
                                    sizeof(int) * min(m, n) * batch_count);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        nsv    = (int*)mem[0];
        V_copy = (rocblas_float_complex*)mem[1];
        ifail  = (int*)mem[2];
    }

    // perform computation
    CHECK_ROCBLAS_ERROR(
        rocsolver_cgesvdx_strided_batched((rocblas_handle)handle,
                                          hip2rocblas_evect2svect(jobz, 1),
                                          hip2rocblas_evect2svect(jobz, 1),
                                          rocblas_srange_index,
                                          m,
                                          n,
                                          (rocblas_float_complex*)const_cast<hipFloatComplex*>(A),
                                          lda,
                                          strideA,
                                          0,
                                          0,
                                          1,
                                          rank,
                                          nsv,
                                          S,
                                          strideS,
                                          (rocblas_float_complex*)U,
                                          ldu,
                                          strideU,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          ifail,
                                          min(m, n),
                                          devInfo,
                                          batch_count));

    // transpose V
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
        return rocblas2hip_status(
            rocblas_cgeam_strided_batched((rocblas_handle)handle,
                                          rocblas_operation_conjugate_transpose,
                                          rocblas_operation_conjugate_transpose,
                                          n,
                                          rank,
                                          &one,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          &zero,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          (rocblas_float_complex*)V,
                                          ldv,
                                          strideV,
                                          batch_count));
    else
        return HIPSOLVER_STATUS_SUCCESS;
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
    if(ldv < 1)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocblas_device_malloc   mem((rocblas_handle)handle);
    int*                    nsv;
    rocblas_double_complex* V_copy;
    int*                    ifail;

    const rocblas_double_complex one         = {1.0f, 0.0f};
    const rocblas_double_complex zero        = {0.0f, 0.0f};
    bool                         use_V_copy  = false;
    int                          ldv_copy    = 1;
    size_t                       size_V_copy = 0;
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
    {
        if(ldv < n || !V)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        use_V_copy  = true;
        ldv_copy    = min(m, n);
        size_V_copy = sizeof(rocblas_double_complex) * ldv_copy * n * batch_count;
    }

    // prepare workspace
    if(work && lwork)
    {
        nsv = (int*)work;
        if(batch_count > 0)
            work = (hipDoubleComplex*)(nsv + batch_count);

        V_copy = (rocblas_double_complex*)work;
        if(use_V_copy)
            work = (hipDoubleComplex*)(V_copy + ldv_copy * n * batch_count);

        ifail = (int*)work;
        if(use_V_copy)
            work = (hipDoubleComplex*)(ifail + n * batch_count);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnZgesvdaStridedBatched_bufferSize((rocblas_handle)handle,
                                                                          jobz,
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
                                                                          &lwork,
                                                                          batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle,
                                    sizeof(int) * batch_count,
                                    size_V_copy,
                                    sizeof(int) * min(m, n) * batch_count);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        nsv    = (int*)mem[0];
        V_copy = (rocblas_double_complex*)mem[1];
        ifail  = (int*)mem[2];
    }

    // perform computation
    CHECK_ROCBLAS_ERROR(
        rocsolver_zgesvdx_strided_batched((rocblas_handle)handle,
                                          hip2rocblas_evect2svect(jobz, 1),
                                          hip2rocblas_evect2svect(jobz, 1),
                                          rocblas_srange_index,
                                          m,
                                          n,
                                          (rocblas_double_complex*)const_cast<hipDoubleComplex*>(A),
                                          lda,
                                          strideA,
                                          0,
                                          0,
                                          1,
                                          rank,
                                          nsv,
                                          S,
                                          strideS,
                                          (rocblas_double_complex*)U,
                                          ldu,
                                          strideU,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          ifail,
                                          min(m, n),
                                          devInfo,
                                          batch_count));

    // transpose V
    if(jobz != HIPSOLVER_EIG_MODE_NOVECTOR)
        return rocblas2hip_status(
            rocblas_zgeam_strided_batched((rocblas_handle)handle,
                                          rocblas_operation_conjugate_transpose,
                                          rocblas_operation_conjugate_transpose,
                                          n,
                                          rank,
                                          &one,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          &zero,
                                          V_copy,
                                          ldv_copy,
                                          ldv_copy * n,
                                          (rocblas_double_complex*)V,
                                          ldv,
                                          strideV,
                                          batch_count));
    else
        return HIPSOLVER_STATUS_SUCCESS;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_sgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
    rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
    rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
    rocsolver_cgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
    rocsolver_zgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSgetrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(devIpiv != nullptr)
        return rocblas2hip_status(
            rocsolver_sgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(
            rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDgetrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(devIpiv != nullptr)
        return rocblas2hip_status(
            rocsolver_dgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(
            rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCgetrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(devIpiv != nullptr)
        return rocblas2hip_status(rocsolver_cgetrf(
            (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(rocsolver_cgetrf_npvt(
            (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZgetrf_bufferSize((rocblas_handle)handle, m, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    if(devIpiv != nullptr)
        return rocblas2hip_status(rocsolver_zgetrf(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(rocsolver_zgetrf_npvt(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devInfo));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_sgetrs((rocblas_handle)handle,
                                                                   hip2rocblas_operation(trans),
                                                                   n,
                                                                   nrhs,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dgetrs((rocblas_handle)handle,
                                                                   hip2rocblas_operation(trans),
                                                                   n,
                                                                   nrhs,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cgetrs((rocblas_handle)handle,
                                                                   hip2rocblas_operation(trans),
                                                                   n,
                                                                   nrhs,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zgetrs((rocblas_handle)handle,
                                                                   hip2rocblas_operation(trans),
                                                                   n,
                                                                   nrhs,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSgetrs_bufferSize(
            (rocblas_handle)handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_sgetrs(
        (rocblas_handle)handle, hip2rocblas_operation(trans), n, nrhs, A, lda, devIpiv, B, ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDgetrs_bufferSize(
            (rocblas_handle)handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_dgetrs(
        (rocblas_handle)handle, hip2rocblas_operation(trans), n, nrhs, A, lda, devIpiv, B, ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCgetrs_bufferSize(
            (rocblas_handle)handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cgetrs((rocblas_handle)handle,
                                               hip2rocblas_operation(trans),
                                               n,
                                               nrhs,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               devIpiv,
                                               (rocblas_float_complex*)B,
                                               ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZgetrs_bufferSize(
            (rocblas_handle)handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zgetrs((rocblas_handle)handle,
                                               hip2rocblas_operation(trans),
                                               n,
                                               nrhs,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               devIpiv,
                                               (rocblas_double_complex*)B,
                                               ldb));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_spotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dpotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cpotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zpotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSpotrf_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_spotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDpotrf_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_dpotrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCpotrf_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_cpotrf((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZpotrf_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zpotrf((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_spotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr, batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dpotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr, batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cpotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr, batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zpotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr, batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSpotrfBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, A, lda, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_spotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo, batch_count));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDpotrfBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, A, lda, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_dpotrf_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo, batch_count));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCpotrfBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, A, lda, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_cpotrf_batched((rocblas_handle)handle,
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       (rocblas_float_complex**)A,
                                                       lda,
                                                       devInfo,
                                                       batch_count));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZpotrfBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, A, lda, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zpotrf_batched((rocblas_handle)handle,
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       (rocblas_double_complex**)A,
                                                       lda,
                                                       devInfo,
                                                       batch_count));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_spotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_dpotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_cpotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(
        rocsolver_zpotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSpotri_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_spotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDpotri_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_dpotri((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCpotri_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_cpotri((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZpotri_bufferSize((rocblas_handle)handle, uplo, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zpotri((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_spotrs(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, nullptr, lda, nullptr, ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dpotrs(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, nullptr, lda, nullptr, ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cpotrs(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, nullptr, lda, nullptr, ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zpotrs(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, nullptr, lda, nullptr, ldb));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSpotrs_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_spotrs((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, A, lda, B, ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDpotrs_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_dpotrs((rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, A, lda, B, ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCpotrs_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_cpotrs((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               nrhs,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)B,
                                               ldb));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZpotrs_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zpotrs((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               nrhs,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)B,
                                               ldb));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_spotrs_batched((rocblas_handle)handle,
                                                                           hip2rocblas_fill(uplo),
                                                                           n,
                                                                           nrhs,
                                                                           nullptr,
                                                                           lda,
                                                                           nullptr,
                                                                           ldb,
                                                                           batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dpotrs_batched((rocblas_handle)handle,
                                                                           hip2rocblas_fill(uplo),
                                                                           n,
                                                                           nrhs,
                                                                           nullptr,
                                                                           lda,
                                                                           nullptr,
                                                                           ldb,
                                                                           batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cpotrs_batched((rocblas_handle)handle,
                                                                           hip2rocblas_fill(uplo),
                                                                           n,
                                                                           nrhs,
                                                                           nullptr,
                                                                           lda,
                                                                           nullptr,
                                                                           ldb,
                                                                           batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zpotrs_batched((rocblas_handle)handle,
                                                                           hip2rocblas_fill(uplo),
                                                                           n,
                                                                           nrhs,
                                                                           nullptr,
                                                                           lda,
                                                                           nullptr,
                                                                           ldb,
                                                                           batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSpotrsBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, batch_count));

    return rocblas2hip_status(rocsolver_spotrs_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, A, lda, B, ldb, batch_count));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDpotrsBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, batch_count));

    return rocblas2hip_status(rocsolver_dpotrs_batched(
        (rocblas_handle)handle, hip2rocblas_fill(uplo), n, nrhs, A, lda, B, ldb, batch_count));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCpotrsBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, batch_count));

    return rocblas2hip_status(rocsolver_cpotrs_batched((rocblas_handle)handle,
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nrhs,
                                                       (rocblas_float_complex**)A,
                                                       lda,
                                                       (rocblas_float_complex**)B,
                                                       ldb,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZpotrsBatched_bufferSize(
            (rocblas_handle)handle, uplo, n, nrhs, A, lda, B, ldb, &lwork, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, batch_count));

    return rocblas2hip_status(rocsolver_zpotrs_batched((rocblas_handle)handle,
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nrhs,
                                                       (rocblas_double_complex**)A,
                                                       lda,
                                                       (rocblas_double_complex**)B,
                                                       ldb,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssyevd((rocblas_handle)handle,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(float) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsyevd((rocblas_handle)handle,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(double) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cheevd((rocblas_handle)handle,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(float) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zheevd((rocblas_handle)handle,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(double) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    float*                E;

    if(work && lwork)
    {
        E = work;
        if(n > 0)
            work = E + n;

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSsyevd_bufferSize((rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (float*)mem[0];
    }

    return rocblas2hip_status(rocsolver_ssyevd((rocblas_handle)handle,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    double*               E;

    if(work && lwork)
    {
        E = work;
        if(n > 0)
            work = E + n;

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDsyevd_bufferSize((rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (double*)mem[0];
    }

    return rocblas2hip_status(rocsolver_dsyevd((rocblas_handle)handle,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    float*                E;

    if(work && lwork)
    {
        E = (float*)work;
        if(n > 0)
            work = (hipFloatComplex*)(E + n);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCheevd_bufferSize((rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (float*)mem[0];
    }

    return rocblas2hip_status(rocsolver_cheevd((rocblas_handle)handle,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    double*               E;

    if(work && lwork)
    {
        E = (double*)work;
        if(n > 0)
            work = (hipDoubleComplex*)(E + n);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZheevd_bufferSize((rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (double*)mem[0];
    }

    return rocblas2hip_status(rocsolver_zheevd((rocblas_handle)handle,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               W,
                                               E,
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
                                                int*                m,
                                                const float*        W,
                                                int*                lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_ssyevdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_dsyevdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_cheevdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_zheevdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnSsyevdx_bufferSize(
            (rocblas_handle)handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_ssyevdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnDsyevdx_bufferSize(
            (rocblas_handle)handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_dsyevdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnCheevdx_bufferSize(
            (rocblas_handle)handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_cheevdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        (rocblas_float_complex*)A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnZheevdx_bufferSize(
            (rocblas_handle)handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zheevdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        (rocblas_double_complex*)A,
                                                        lda,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssyevj((rocblas_handle)handle,
                                                                   rocblas_esort_ascending,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsyevj((rocblas_handle)handle,
                                                                   rocblas_esort_ascending,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cheevj((rocblas_handle)handle,
                                                                   rocblas_esort_ascending,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zheevj((rocblas_handle)handle,
                                                                   rocblas_esort_ascending,
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSsyevj_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_ssyevj((rocblas_handle)handle,
                                               rocblas_esort_ascending,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               info->tolerance,
                                               (float*)info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDsyevj_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_dsyevj((rocblas_handle)handle,
                                               rocblas_esort_ascending,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               info->tolerance,
                                               info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCheevj_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_cheevj((rocblas_handle)handle,
                                               rocblas_esort_ascending,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               info->tolerance,
                                               (float*)info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZheevj_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_zheevj((rocblas_handle)handle,
                                               rocblas_esort_ascending,
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               info->tolerance,
                                               info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssyevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        nullptr,
        lda,
        lda * n,
        info->tolerance,
        nullptr,
        info->max_sweeps,
        nullptr,
        nullptr,
        n,
        nullptr,
        batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsyevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        nullptr,
        lda,
        lda * n,
        info->tolerance,
        nullptr,
        info->max_sweeps,
        nullptr,
        nullptr,
        n,
        nullptr,
        batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_cheevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        nullptr,
        lda,
        lda * n,
        info->tolerance,
        nullptr,
        info->max_sweeps,
        nullptr,
        nullptr,
        n,
        nullptr,
        batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zheevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        nullptr,
        lda,
        lda * n,
        info->tolerance,
        nullptr,
        info->max_sweeps,
        nullptr,
        nullptr,
        n,
        nullptr,
        batch_count));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSsyevjBatched_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_ssyevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        A,
        lda,
        lda * n,
        info->tolerance,
        (float*)info->residual,
        info->max_sweeps,
        info->n_sweeps,
        W,
        n,
        devInfo,
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDsyevjBatched_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_dsyevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        A,
        lda,
        lda * n,
        info->tolerance,
        info->residual,
        info->max_sweeps,
        info->n_sweeps,
        W,
        n,
        devInfo,
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverCheevjBatched_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_cheevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        (rocblas_float_complex*)A,
        lda,
        lda * n,
        info->tolerance,
        (float*)info->residual,
        info->max_sweeps,
        info->n_sweeps,
        W,
        n,
        devInfo,
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZheevjBatched_bufferSize(
            (rocblas_handle)handle, jobz, uplo, n, A, lda, W, &lwork, info, batch_count));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(batch_count));
    info->is_batched = true;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_zheevj_strided_batched(
        (rocblas_handle)handle,
        (info->sort_eig ? rocblas_esort_ascending : rocblas_esort_none),
        hip2rocblas_evect(jobz),
        hip2rocblas_fill(uplo),
        n,
        (rocblas_double_complex*)A,
        lda,
        lda * n,
        info->tolerance,
        info->residual,
        info->max_sweeps,
        info->n_sweeps,
        W,
        n,
        devInfo,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssygvd((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(float) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsygvd((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(double) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_chegvd((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(float) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zhegvd((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // space for E array
    size_t size_E = n > 0 ? sizeof(double) * n : 0;

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    float*                E;

    if(work && lwork)
    {
        E = work;
        if(n > 0)
            work = E + n;

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSsygvd_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (float*)mem[0];
    }

    return rocblas2hip_status(rocsolver_ssygvd((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               B,
                                               ldb,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    double*               E;

    if(work && lwork)
    {
        E = work;
        if(n > 0)
            work = E + n;

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDsygvd_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (double*)mem[0];
    }

    return rocblas2hip_status(rocsolver_dsygvd((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               B,
                                               ldb,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    float*                E;

    if(work && lwork)
    {
        E = (float*)work;
        if(n > 0)
            work = (hipFloatComplex*)(E + n);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverChegvd_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(float) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (float*)mem[0];
    }

    return rocblas2hip_status(rocsolver_chegvd((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)B,
                                               ldb,
                                               W,
                                               E,
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
    rocblas_device_malloc mem((rocblas_handle)handle);
    double*               E;

    if(work && lwork)
    {
        E = (double*)work;
        if(n > 0)
            work = (hipDoubleComplex*)(E + n);

        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    }
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZhegvd_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

        mem = rocblas_device_malloc((rocblas_handle)handle, sizeof(double) * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (double*)mem[0];
    }

    return rocblas2hip_status(rocsolver_zhegvd((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)B,
                                               ldb,
                                               W,
                                               E,
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_ssygvdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_eform(itype),
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       nullptr,
                                                       ldb,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_dsygvdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_eform(itype),
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       nullptr,
                                                       ldb,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_chegvdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_eform(itype),
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       nullptr,
                                                       ldb,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status
        = rocblas2hip_status(rocsolver_zhegvdx_inplace((rocblas_handle)handle,
                                                       hip2rocblas_eform(itype),
                                                       hip2rocblas_evect(jobz),
                                                       hip2rocblas_erange(range),
                                                       hip2rocblas_fill(uplo),
                                                       n,
                                                       nullptr,
                                                       lda,
                                                       nullptr,
                                                       ldb,
                                                       vl,
                                                       vu,
                                                       il,
                                                       iu,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnSsygvdx_bufferSize((rocblas_handle)handle,
                                                            itype,
                                                            jobz,
                                                            range,
                                                            uplo,
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
                                                            &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_ssygvdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_eform(itype),
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnDsygvdx_bufferSize((rocblas_handle)handle,
                                                            itype,
                                                            jobz,
                                                            range,
                                                            uplo,
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
                                                            &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_dsygvdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_eform(itype),
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        A,
                                                        lda,
                                                        B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnChegvdx_bufferSize((rocblas_handle)handle,
                                                            itype,
                                                            jobz,
                                                            range,
                                                            uplo,
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
                                                            &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_chegvdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_eform(itype),
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        (rocblas_float_complex*)A,
                                                        lda,
                                                        (rocblas_float_complex*)B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnZhegvdx_bufferSize((rocblas_handle)handle,
                                                            itype,
                                                            jobz,
                                                            range,
                                                            uplo,
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
                                                            &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zhegvdx_inplace((rocblas_handle)handle,
                                                        hip2rocblas_eform(itype),
                                                        hip2rocblas_evect(jobz),
                                                        hip2rocblas_erange(range),
                                                        hip2rocblas_fill(uplo),
                                                        n,
                                                        (rocblas_double_complex*)A,
                                                        lda,
                                                        (rocblas_double_complex*)B,
                                                        ldb,
                                                        vl,
                                                        vu,
                                                        il,
                                                        iu,
                                                        0,
                                                        nev,
                                                        W,
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssygvj((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsygvj((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_chegvj((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zhegvj((rocblas_handle)handle,
                                                                   hip2rocblas_eform(itype),
                                                                   hip2rocblas_evect(jobz),
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   ldb,
                                                                   info->tolerance,
                                                                   nullptr,
                                                                   info->max_sweeps,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverSsygvj_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(
            hipsolverManageWorkspace((rocblas_handle)handle, lwork + sizeof(float) * n));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_ssygvj((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               B,
                                               ldb,
                                               info->tolerance,
                                               (float*)info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDsygvj_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(
            hipsolverManageWorkspace((rocblas_handle)handle, lwork + sizeof(float) * n));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_dsygvj((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               A,
                                               lda,
                                               B,
                                               ldb,
                                               info->tolerance,
                                               info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverChegvj_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(
            hipsolverManageWorkspace((rocblas_handle)handle, lwork + sizeof(float) * n));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = true;

    return rocblas2hip_status(rocsolver_chegvj((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               (rocblas_float_complex*)B,
                                               ldb,
                                               info->tolerance,
                                               (float*)info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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

    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverZhegvj_bufferSize(
            (rocblas_handle)handle, itype, jobz, uplo, n, A, lda, B, ldb, W, &lwork, info));
        CHECK_ROCBLAS_ERROR(
            hipsolverManageWorkspace((rocblas_handle)handle, lwork + sizeof(float) * n));
    }

    CHECK_HIPSOLVER_ERROR(info->setup(1));
    info->is_batched = false;
    info->is_float   = false;

    return rocblas2hip_status(rocsolver_zhegvj((rocblas_handle)handle,
                                               hip2rocblas_eform(itype),
                                               hip2rocblas_evect(jobz),
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               (rocblas_double_complex*)B,
                                               ldb,
                                               info->tolerance,
                                               info->residual,
                                               info->max_sweeps,
                                               info->n_sweeps,
                                               W,
                                               devInfo));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssytrd((rocblas_handle)handle,
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsytrd((rocblas_handle)handle,
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_chetrd((rocblas_handle)handle,
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zhetrd((rocblas_handle)handle,
                                                                   hip2rocblas_fill(uplo),
                                                                   n,
                                                                   nullptr,
                                                                   lda,
                                                                   nullptr,
                                                                   nullptr,
                                                                   nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSsytrd_bufferSize((rocblas_handle)handle, uplo, n, A, lda, D, E, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_ssytrd((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, D, E, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDsytrd_bufferSize((rocblas_handle)handle, uplo, n, A, lda, D, E, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(
        rocsolver_dsytrd((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, D, E, tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverChetrd_bufferSize((rocblas_handle)handle, uplo, n, A, lda, D, E, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_chetrd((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               D,
                                               E,
                                               (rocblas_float_complex*)tau));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZhetrd_bufferSize((rocblas_handle)handle, uplo, n, A, lda, D, E, tau, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    return rocblas2hip_status(rocsolver_zhetrd((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               D,
                                               E,
                                               (rocblas_double_complex*)tau));
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_ssytrf(
        (rocblas_handle)handle, rocblas_fill_upper, n, nullptr, lda, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_dsytrf(
        (rocblas_handle)handle, rocblas_fill_upper, n, nullptr, lda, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_csytrf(
        (rocblas_handle)handle, rocblas_fill_upper, n, nullptr, lda, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(lwork == nullptr)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;
    size_t sz;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status = rocblas2hip_status(rocsolver_zsytrf(
        (rocblas_handle)handle, rocblas_fill_upper, n, nullptr, lda, nullptr, nullptr));
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;
    if(sz > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    *lwork = (int)sz;
    return status;
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverSsytrf_bufferSize((rocblas_handle)handle, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_ssytrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, ipiv, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverDsytrf_bufferSize((rocblas_handle)handle, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(
        rocsolver_dsytrf((rocblas_handle)handle, hip2rocblas_fill(uplo), n, A, lda, ipiv, devInfo));
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverCsytrf_bufferSize((rocblas_handle)handle, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_csytrf((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_float_complex*)A,
                                               lda,
                                               ipiv,
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
    if(work && lwork)
        CHECK_ROCBLAS_ERROR(rocblas_set_workspace((rocblas_handle)handle, work, lwork));
    else
    {
        CHECK_HIPSOLVER_ERROR(
            hipsolverZsytrf_bufferSize((rocblas_handle)handle, n, A, lda, &lwork));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));
    }

    return rocblas2hip_status(rocsolver_zsytrf((rocblas_handle)handle,
                                               hip2rocblas_fill(uplo),
                                               n,
                                               (rocblas_double_complex*)A,
                                               lda,
                                               ipiv,
                                               devInfo));
}
catch(...)
{
    return exception2hip_status();
}

} // extern C
