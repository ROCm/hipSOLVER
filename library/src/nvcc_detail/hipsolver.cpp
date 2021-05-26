/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsolver.h"
#include "exceptions.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <hip/hip_runtime.h>

extern "C" {

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
        throw HIPSOLVER_STATUS_INVALID_ENUM;
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
        throw HIPSOLVER_STATUS_INVALID_ENUM;
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
        throw HIPSOLVER_STATUS_INVALID_ENUM;
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
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cublasSideMode_t hip2cuda_side(hipsolverSideMode_t side)
{
    switch(side)
    {
    case HIPSOLVER_SIDE_LEFT:
        return CUBLAS_SIDE_LEFT;
    case HIPSOLVER_SIDE_RIGHT:
        return CUBLAS_SIDE_RIGHT;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverSideMode_t cuda2hip_side(cublasSideMode_t side)
{
    switch(side)
    {
    case CUBLAS_SIDE_LEFT:
        return HIPSOLVER_SIDE_LEFT;
    case CUBLAS_SIDE_RIGHT:
        return HIPSOLVER_SIDE_RIGHT;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

cusolverEigMode_t hip2cuda_evect(hipsolverEigMode_t eig)
{
    switch(eig)
    {
    case HIPSOLVER_EIG_MODE_NOVECTOR:
        return CUSOLVER_EIG_MODE_NOVECTOR;
    case HIPSOLVER_EIG_MODE_VECTOR:
        return CUSOLVER_EIG_MODE_VECTOR;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
    }
}

hipsolverEigMode_t cuda2hip_evect(cusolverEigMode_t eig)
{
    switch(eig)
    {
    case CUSOLVER_EIG_MODE_NOVECTOR:
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    case CUSOLVER_EIG_MODE_VECTOR:
        return HIPSOLVER_EIG_MODE_VECTOR;
    default:
        throw HIPSOLVER_STATUS_INVALID_ENUM;
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
try
{
    return cuda2hip_status(cusolverDnCreate((cusolverDnHandle_t*)handle));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle)
try
{
    return cuda2hip_status(cusolverDnDestroy((cusolverDnHandle_t)handle));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle, hipStream_t streamId)
try
{
    return cuda2hip_status(cusolverDnSetStream((cusolverDnHandle_t)handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle, hipStream_t* streamId)
try
{
    return cuda2hip_status(cusolverDnGetStream((cusolverDnHandle_t)handle, streamId));
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
    return cuda2hip_status(
        cusolverDnSorgqr_bufferSize((cusolverDnHandle_t)handle, m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnDorgqr_bufferSize((cusolverDnHandle_t)handle, m, n, k, A, lda, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle,
                                             int               m,
                                             int               n,
                                             int               k,
                                             hipsolverComplex* A,
                                             int               lda,
                                             hipsolverComplex* tau,
                                             int*              lwork)
try
{
    return cuda2hip_status(cusolverDnCungqr_bufferSize(
        (cusolverDnHandle_t)handle, m, n, k, (cuComplex*)A, lda, (cuComplex*)tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t       handle,
                                             int                     m,
                                             int                     n,
                                             int                     k,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* tau,
                                             int*                    lwork)
try
{
    return cuda2hip_status(cusolverDnZungqr_bufferSize((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(
        cusolverDnSorgqr((cusolverDnHandle_t)handle, m, n, k, A, lda, tau, work, lwork, devInfo));
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
    return cuda2hip_status(
        cusolverDnDorgqr((cusolverDnHandle_t)handle, m, n, k, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* tau,
                                  hipsolverComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    return cuda2hip_status(cusolverDnCungqr((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* tau,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZungqr((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSormqr_bufferSize((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnDormqr_bufferSize((cusolverDnHandle_t)handle,
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
                                             hipsolverComplex*    A,
                                             int                  lda,
                                             hipsolverComplex*    tau,
                                             hipsolverComplex*    C,
                                             int                  ldc,
                                             int*                 lwork)
try
{
    return cuda2hip_status(cusolverDnCunmqr_bufferSize((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t       handle,
                                             hipsolverSideMode_t     side,
                                             hipsolverOperation_t    trans,
                                             int                     m,
                                             int                     n,
                                             int                     k,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             hipsolverDoubleComplex* tau,
                                             hipsolverDoubleComplex* C,
                                             int                     ldc,
                                             int*                    lwork)
try
{
    return cuda2hip_status(cusolverDnZunmqr_bufferSize((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSormqr((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnDormqr((cusolverDnHandle_t)handle,
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
                                  hipsolverComplex*    A,
                                  int                  lda,
                                  hipsolverComplex*    tau,
                                  hipsolverComplex*    C,
                                  int                  ldc,
                                  hipsolverComplex*    work,
                                  int                  lwork,
                                  int*                 devInfo)
try
{
    return cuda2hip_status(cusolverDnCunmqr((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t       handle,
                                  hipsolverSideMode_t     side,
                                  hipsolverOperation_t    trans,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* tau,
                                  hipsolverDoubleComplex* C,
                                  int                     ldc,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZunmqr((cusolverDnHandle_t)handle,
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

/******************** GEBRD ********************/
hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    return cuda2hip_status(cusolverDnSgebrd_bufferSize((cusolverDnHandle_t)handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    return cuda2hip_status(cusolverDnDgebrd_bufferSize((cusolverDnHandle_t)handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    return cuda2hip_status(cusolverDnCgebrd_bufferSize((cusolverDnHandle_t)handle, m, n, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
try
{
    return cuda2hip_status(cusolverDnZgebrd_bufferSize((cusolverDnHandle_t)handle, m, n, lwork));
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
    return cuda2hip_status(cusolverDnSgebrd(
        (cusolverDnHandle_t)handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo));
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
    return cuda2hip_status(cusolverDnDgebrd(
        (cusolverDnHandle_t)handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  float*            D,
                                  float*            E,
                                  hipsolverComplex* tauq,
                                  hipsolverComplex* taup,
                                  hipsolverComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    return cuda2hip_status(cusolverDnCgebrd((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  double*                 D,
                                  double*                 E,
                                  hipsolverDoubleComplex* tauq,
                                  hipsolverDoubleComplex* taup,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZgebrd((cusolverDnHandle_t)handle,
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

/******************** GEQRF ********************/
hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnSgeqrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnDgeqrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnCgeqrf_bufferSize((cusolverDnHandle_t)handle, m, n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
try
{
    return cuda2hip_status(cusolverDnZgeqrf_bufferSize(
        (cusolverDnHandle_t)handle, m, n, (cuDoubleComplex*)A, lda, lwork));
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
    return cuda2hip_status(
        cusolverDnSgeqrf((cusolverDnHandle_t)handle, m, n, A, lda, tau, work, lwork, devInfo));
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
    return cuda2hip_status(
        cusolverDnDgeqrf((cusolverDnHandle_t)handle, m, n, A, lda, tau, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* tau,
                                  hipsolverComplex* work,
                                  int               lwork,
                                  int*              devInfo)
try
{
    return cuda2hip_status(cusolverDnCgeqrf((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* tau,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZgeqrf((cusolverDnHandle_t)handle,
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

/******************** GETRF ********************/
hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnSgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnDgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
try
{
    return cuda2hip_status(
        cusolverDnCgetrf_bufferSize((cusolverDnHandle_t)handle, m, n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
try
{
    return cuda2hip_status(cusolverDnZgetrf_bufferSize(
        (cusolverDnHandle_t)handle, m, n, (cuDoubleComplex*)A, lda, lwork));
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
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    return cuda2hip_status(
        cusolverDnSgetrf((cusolverDnHandle_t)handle, m, n, A, lda, work, devIpiv, devInfo));
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
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    return cuda2hip_status(
        cusolverDnDgetrf((cusolverDnHandle_t)handle, m, n, A, lda, work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                  int               m,
                                  int               n,
                                  hipsolverComplex* A,
                                  int               lda,
                                  hipsolverComplex* work,
                                  int*              devIpiv,
                                  int*              devInfo)
try
{
    return cuda2hip_status(cusolverDnCgetrf(
        (cusolverDnHandle_t)handle, m, n, (cuComplex*)A, lda, (cuComplex*)work, devIpiv, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* work,
                                  int*                    devIpiv,
                                  int*                    devInfo)
try
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
catch(...)
{
    return exception2hip_status();
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  float*               A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  float*               B,
                                  int                  ldb,
                                  int*                 devInfo)
try
{
    return cuda2hip_status(cusolverDnSgetrs((cusolverDnHandle_t)handle,
                                            hip2cuda_operation(trans),
                                            n,
                                            nrhs,
                                            A,
                                            lda,
                                            devIpiv,
                                            B,
                                            ldb,
                                            devInfo));
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
                                  int*                 devInfo)
try
{
    return cuda2hip_status(cusolverDnDgetrs((cusolverDnHandle_t)handle,
                                            hip2cuda_operation(trans),
                                            n,
                                            nrhs,
                                            A,
                                            lda,
                                            devIpiv,
                                            B,
                                            ldb,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                  hipsolverOperation_t trans,
                                  int                  n,
                                  int                  nrhs,
                                  hipsolverComplex*    A,
                                  int                  lda,
                                  int*                 devIpiv,
                                  hipsolverComplex*    B,
                                  int                  ldb,
                                  int*                 devInfo)
try
{
    return cuda2hip_status(cusolverDnCgetrs((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t       handle,
                                  hipsolverOperation_t    trans,
                                  int                     n,
                                  int                     nrhs,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  int*                    devIpiv,
                                  hipsolverDoubleComplex* B,
                                  int                     ldb,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZgetrs((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSpotrf_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
try
{
    return cuda2hip_status(cusolverDnDpotrf_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipsolverComplex*   A,
                                             int                 lda,
                                             int*                lwork)
try
{
    return cuda2hip_status(cusolverDnCpotrf_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, (cuComplex*)A, lda, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t       handle,
                                             hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             int*                    lwork)
try
{
    return cuda2hip_status(cusolverDnZpotrf_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, (cuDoubleComplex*)A, lda, lwork));
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
    return cuda2hip_status(cusolverDnSpotrf(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
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
    return cuda2hip_status(cusolverDnDpotrf(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, work, lwork, devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipsolverComplex*   A,
                                  int                 lda,
                                  hipsolverComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    return cuda2hip_status(cusolverDnCpotrf((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t       handle,
                                  hipsolverFillMode_t     uplo,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZpotrf((cusolverDnHandle_t)handle,
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
hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    return cuda2hip_status(cusolverDnSpotrfBatched(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, devInfo, batch_count));
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
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    return cuda2hip_status(cusolverDnDpotrfBatched(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, devInfo, batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int*                devInfo,
                                         int                 batch_count)
try
{
    return cuda2hip_status(cusolverDnCpotrfBatched((cusolverDnHandle_t)handle,
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   (cuComplex**)A,
                                                   lda,
                                                   devInfo,
                                                   batch_count));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int*                    devInfo,
                                         int                     batch_count)
try
{
    return cuda2hip_status(cusolverDnZpotrfBatched((cusolverDnHandle_t)handle,
                                                   hip2cuda_fill(uplo),
                                                   n,
                                                   (cuDoubleComplex**)A,
                                                   lda,
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
    return cuda2hip_status(cusolverDnSsyevd_bufferSize((cusolverDnHandle_t)handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       W,
                                                       lwork));
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
    return cuda2hip_status(cusolverDnDsyevd_bufferSize((cusolverDnHandle_t)handle,
                                                       hip2cuda_evect(jobz),
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       A,
                                                       lda,
                                                       W,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverEigMode_t  jobz,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipsolverComplex*   A,
                                             int                 lda,
                                             float*              W,
                                             int*                lwork)
try
{
    return cuda2hip_status(cusolverDnCheevd_bufferSize((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t       handle,
                                             hipsolverEigMode_t      jobz,
                                             hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             double*                 W,
                                             int*                    lwork)
try
{
    return cuda2hip_status(cusolverDnZheevd_bufferSize((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSsyevd((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnDsyevd((cusolverDnHandle_t)handle,
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
                                  hipsolverComplex*   A,
                                  int                 lda,
                                  float*              W,
                                  hipsolverComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    return cuda2hip_status(cusolverDnCheevd((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t       handle,
                                  hipsolverEigMode_t      jobz,
                                  hipsolverFillMode_t     uplo,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  double*                 W,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZheevd((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSsytrd_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, lwork));
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
    return cuda2hip_status(cusolverDnDsytrd_bufferSize(
        (cusolverDnHandle_t)handle, hip2cuda_fill(uplo), n, A, lda, D, E, tau, lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipsolverComplex*   A,
                                             int                 lda,
                                             float*              D,
                                             float*              E,
                                             hipsolverComplex*   tau,
                                             int*                lwork)
try
{
    return cuda2hip_status(cusolverDnChetrd_bufferSize((cusolverDnHandle_t)handle,
                                                       hip2cuda_fill(uplo),
                                                       n,
                                                       (cuComplex*)A,
                                                       lda,
                                                       D,
                                                       E,
                                                       (cuComplex*)tau,
                                                       lwork));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t       handle,
                                             hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A,
                                             int                     lda,
                                             double*                 D,
                                             double*                 E,
                                             hipsolverDoubleComplex* tau,
                                             int*                    lwork)
try
{
    return cuda2hip_status(cusolverDnZhetrd_bufferSize((cusolverDnHandle_t)handle,
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
    return cuda2hip_status(cusolverDnSsytrd((cusolverDnHandle_t)handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            D,
                                            E,
                                            tau,
                                            work,
                                            lwork,
                                            devInfo));
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
    return cuda2hip_status(cusolverDnDsytrd((cusolverDnHandle_t)handle,
                                            hip2cuda_fill(uplo),
                                            n,
                                            A,
                                            lda,
                                            D,
                                            E,
                                            tau,
                                            work,
                                            lwork,
                                            devInfo));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t   handle,
                                  hipsolverFillMode_t uplo,
                                  int                 n,
                                  hipsolverComplex*   A,
                                  int                 lda,
                                  float*              D,
                                  float*              E,
                                  hipsolverComplex*   tau,
                                  hipsolverComplex*   work,
                                  int                 lwork,
                                  int*                devInfo)
try
{
    return cuda2hip_status(cusolverDnChetrd((cusolverDnHandle_t)handle,
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

hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t       handle,
                                  hipsolverFillMode_t     uplo,
                                  int                     n,
                                  hipsolverDoubleComplex* A,
                                  int                     lda,
                                  double*                 D,
                                  double*                 E,
                                  hipsolverDoubleComplex* tau,
                                  hipsolverDoubleComplex* work,
                                  int                     lwork,
                                  int*                    devInfo)
try
{
    return cuda2hip_status(cusolverDnZhetrd((cusolverDnHandle_t)handle,
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

} // extern C
