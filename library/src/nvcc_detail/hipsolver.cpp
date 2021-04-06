/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cublasOperation_t hip2cuda_operation(hipsolverOperation_t op)
{
    switch(op)
    {
    case HIPSOLVER_OP_N:
        return CUBLAS_OP_N;
    case HIPSOLVER_OP_T:
        return CUBLAS_OP_T;
    case HIPSOLVER_OP_C:
        return CUBLAS_OP_C;
    default:
        throw std::invalid_argument("Non existent OP");
    }
}

hipsolverOperation_t cuda2hip_operation(cublasOperation_t op)
{
    switch(op)
    {
    case CUBLAS_OP_N:
        return HIPSOLVER_OP_N;
    case CUBLAS_OP_T:
        return HIPSOLVER_OP_T;
    case CUBLAS_OP_C:
        return HIPSOLVER_OP_C;
    default:
        throw std::invalid_argument("Non existent OP");
    }
}

cublasFillMode_t hip2cuda_fill(hipsolverFillMode_t fill)
{
    switch(fill)
    {
    case HIPSOLVER_FILL_MODE_UPPER:
        return CUBLAS_FILL_MODE_UPPER;
    case HIPSOLVER_FILL_MODE_LOWER:
        return CUBLAS_FILL_MODE_LOWER;
    default:
        throw std::invalid_argument("Non existent FILL");
    }
}

hipsolverFillMode_t cuda2hip_fill(cublasFillMode_t fill)
{
    switch(fill)
    {
    case CUBLAS_FILL_MODE_UPPER:
        return HIPSOLVER_FILL_MODE_UPPER;
    case CUBLAS_FILL_MODE_LOWER:
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        throw std::invalid_argument("Non existent FILL");
    }
}

hipsolverStatus_t cuda2hip_status(cusolverStatus_t cuStatus)
{
    switch(cuStatus)
    {
    case CUSOLVER_STATUS_SUCCESS:
        return HIPSOLVER_STATUS_SUCCESS;
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return HIPSOLVER_STATUS_ALLOC_FAILED;
    case CUSOLVER_STATUS_INVALID_VALUE:
        return HIPSOLVER_STATUS_INVALID_VALUE;
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return HIPSOLVER_STATUS_MAPPING_ERROR;
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return HIPSOLVER_STATUS_EXECUTION_FAILED;
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return HIPSOLVER_STATUS_ARCH_MISMATCH;
    default:
        return HIPSOLVER_STATUS_UNKNOWN;
    }
}

hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle)
{
    return cuda2hip_status(cusolverDnCreate((cusolverDnHandle_t*)handle));
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
{
    return cuda2hip_status(cusolverDnDestroy((cusolverDnHandle_t)handle));
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)
{
    return cuda2hip_status(cusolverDnSetStream((cusolverDnHandle_t)handle, streamId));
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId)
{
    return cuda2hip_status(cusolverDnGetStream((cusolverDnHandle_t)handle, streamId));
}

/******************** GETRF ********************/
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    return cuda2hip_status(
        cusolverDnSgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    return cuda2hip_status(
        cusolverDnDgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    return cuda2hip_status(
        cusolverDnCgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, (cuComplex*)A, lda, lwork));
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{
    return cuda2hip_status(cusolverDnZgetrf_bufferSize(
        (cusolverDnHandle_t)handle, m, n, (cuDoubleComplex*)A, lda, lwork));
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
    return cuda2hip_status(
        cusolverDnSgetrf((cusolverDnHandle_t)handle, m, n, A, lda, work, devIpiv, devInfo));
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
    return cuda2hip_status(
        cusolverDnDgetrf((cusolverDnHandle_t)handle, m, n, A, lda, work, devIpiv, devInfo));
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
    return cuda2hip_status(cusolverDnCgetrf(
        (cusolverDnHandle_t)handle, m, n, (cuComplex*)A, lda, (cuComplex*)work, devIpiv, devInfo));
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
    return cuda2hip_status(cusolverDnZgetrf((cusolverDnHandle_t)handle,
                                            m,
                                            n,
                                            (cuDoubleComplex*)A,
                                            lda,
                                            (cuDoubleComplex*)work,
                                            devIpiv,
                                            devInfo));
}

#ifdef __cplusplus
}
#endif
