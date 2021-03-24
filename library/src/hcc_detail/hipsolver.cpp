/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver.h"
#include "rocblas.h"
#include "rocsolver.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <math.h>

using namespace std;

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
    default:
        throw std::invalid_argument("Non existent OP");
    }
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
    default:
        throw std::invalid_argument("Non existent OP");
    }
}

rocblas_fill_ hip2rocblas_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return rocblas_fill_upper;
    case HIPSOLVER_FILL_MODE_LOWER:
        return rocblas_fill_lower;
    default:
        throw std::invalid_argument("Non existent FILL");
    }
}

hipsolverFillMode_t rocblas2hip_fill(rocblas_fill_ fill)
{
    switch(fill)
    {
    case rocblas_fill_upper:
        return HIPSOLVER_FILL_MODE_UPPER;
    case rocblas_fill_lower:
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw std::invalid_argument("Non existent FILL");
    }
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
    default:
        throw std::invalid_argument("Unimplemented status");
    }
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

/******************** GETRF ********************/
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_sgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_dgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_cgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_cgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{
    size_t sz1, sz2;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_status status
        = rocsolver_zgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz1);

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocsolver_zgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz2);

    *lwork = (int)max(sz1, sz2);
    return rocblas2hip_status(status);
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
    if(work != nullptr)
    {
        size_t sz;
        rocblas_start_device_memory_size_query((rocblas_handle)handle);
        if(devIpiv != nullptr)
            rocsolver_sgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
        else
            rocsolver_sgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
        rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

        rocblas_set_workspace((rocblas_handle)handle, (void*)work, sz);
    }

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
    if(work != nullptr)
    {
        size_t sz;
        rocblas_start_device_memory_size_query((rocblas_handle)handle);
        if(devIpiv != nullptr)
            rocsolver_dgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
        else
            rocsolver_dgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
        rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

        rocblas_set_workspace((rocblas_handle)handle, (void*)work, sz);
    }

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
    if(work != nullptr)
    {
        size_t sz;
        rocblas_start_device_memory_size_query((rocblas_handle)handle);
        if(devIpiv != nullptr)
            rocsolver_cgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
        else
            rocsolver_cgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
        rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

        rocblas_set_workspace((rocblas_handle)handle, (void*)work, sz);
    }

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
    if(work != nullptr)
    {
        size_t sz;
        rocblas_start_device_memory_size_query((rocblas_handle)handle);
        if(devIpiv != nullptr)
            rocsolver_zgetrf((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr);
        else
            rocsolver_zgetrf_npvt((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
        rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

        rocblas_set_workspace((rocblas_handle)handle, (void*)work, sz);
    }

    if(devIpiv != nullptr)
        return rocblas2hip_status(rocsolver_zgetrf(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devIpiv, devInfo));
    else
        return rocblas2hip_status(rocsolver_zgetrf_npvt(
            (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devInfo));
}
}
