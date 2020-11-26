/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver.h"
#include "rocblas.h"
#include "rocsolver.h"
#include <algorithm>
#include <functional>
#include <math.h>

extern "C" {

rocblas_operation_ hip2rocblas_operation(hipsolverOperation_t op)
{
    switch(op)
    {
    case HIPSOLVER_OP_N:
        return rocblas_operation_none;
    case HIPSOLVER_OP_T:
        return rocblas_operation_transpose;
    case HIPSOLVER_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    throw "Non existent OP";
}

hipsolverOperation_t rocblas2hip_operation(rocblas_operation_ op)
{
    switch(op)
    {
    case rocblas_operation_none:
        return HIPSOLVER_OP_N;
    case rocblas_operation_transpose:
        return HIPSOLVER_OP_T;
    case rocblas_operation_conjugate_transpose:
        return HIPSOLVER_OP_C;
    }
    throw "Non existent OP";
}

rocblas_fill_ hip2rocblas_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return rocblas_fill_upper;
    case HIPSOLVER_FILL_MODE_LOWER:
        return rocblas_fill_lower;
    case HIPSOLVER_FILL_MODE_FULL:
        return rocblas_fill_full;
    }
    throw "Non existent FILL";
}

hipsolverFillMode_t rocblas2hip_fill(rocblas_fill_ fill)
{
    switch(fill)
    {
    case rocblas_fill_upper:
        return HIPSOLVER_FILL_MODE_UPPER;
    case rocblas_fill_lower:
        return HIPSOLVER_FILL_MODE_LOWER;
    case rocblas_fill_full:
        return HIPSOLVER_FILL_MODE_FULL;
    }
    throw "Non existent FILL";
}

hipsolverStatus_t rocblas2hip_status(rocblas_status_ error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPSOLVER_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case rocblas_status_invalid_size:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    throw "Unimplemented status";
}

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
{
    if(!handle)
        return HIPSOLVER_STATUS_HANDLE_IS_NULLPTR;

    // Create the rocBLAS handle
    return rocblas2hip_status(rocblas_create_handle((rocblas_handle*)handle));
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
{
    return rocblas2hip_status(rocblas_destroy_handle((rocblas_handle)handle));
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return rocblas2hip_status(rocblas_set_stream((rocblas_handle)handle, streamId));
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId)
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return rocblas2hip_status(rocblas_get_stream((rocblas_handle)handle, streamId));
}

// getrf
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{
    *lwork = 0;
    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  float*            A,
                                  int               lda,
                                  float*            work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
        return rocblas2hip_status(
            rocsolver_sgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(
            rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
}

hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  double*           A,
                                  int               lda,
                                  double*           work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
        return rocblas2hip_status(
            rocsolver_dgetrf((rocblas_handle)handle, m, n, A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(
            rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, A, lda, devInfo));
}

hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* work,
                                  int*              devIpiv,
                                  int*              devInfo)
{
    if(devIpiv != nullptr)
        return rocblas2hip_status(rocsolver_cgetrf(
            (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(rocsolver_cgetrf_npvt(
            (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devInfo));
}

hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* work,
                                  int*                    devIpiv,
                                  int*                    devInfo)
{
    if(devIpiv != nullptr)
        return rocblas2hip_status(rocsolver_zgetrf(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(rocsolver_zgetrf_npvt(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devInfo));
}
}
