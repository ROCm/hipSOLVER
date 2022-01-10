/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver_compat_common.cpp provides implementation of the compatibility APIs 
 *  that are equivalent to both, cuSOLVER and rocSOLVER. These simply call hipSOLVER's
 *  regular APIs.   
 */

#include "hipsolver.h"

extern "C" {

// helpers
hipsolverStatus_t hipsolverDnCreate(hipsolverDnHandle_t* handle)
{
    return hipsolverCreate(handle);
}

hipsolverStatus_t hipsolverDnDestroy(hipsolverDnHandle_t handle)
{
    return hipsolverDestroy(handle);
}

hipsolverStatus_t hipsolverDnSetStream(hipsolverDnHandle_t handle, hipStream_t streamId)
{
    return hipsolverSetStream(handle, streamId);
}

hipsolverStatus_t hipsolverDnGetStream(hipsolverDnHandle_t handle, hipStream_t* streamId)
{
    return hipsolverGetStream(handle, streamId);
}

// orgbr/ungbr
hipsolverStatus_t hipsolverDnSorgbr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               int*                lwork)
{
    return hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnDorgbr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               int*                lwork)
{
    return hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnCungbr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               hipFloatComplex*    tau,
                                               int*                lwork)
{
    return hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnZungbr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               hipDoubleComplex*   tau,
                                               int*                lwork)
{
    return hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnSorgbr(hipsolverDnHandle_t handle,
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
{
    return hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDorgbr(hipsolverDnHandle_t handle,
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
{
    return hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCungbr(hipsolverDnHandle_t handle,
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
{
    return hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZungbr(hipsolverDnHandle_t handle,
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
{
    return hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo);
}

// orgqr/ungqr
hipsolverStatus_t hipsolverDnSorgqr_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork)
{
    return hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnDorgqr_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork)
{
    return hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               hipFloatComplex*    tau,
                                               int*                lwork)
{
    return hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               hipDoubleComplex*   tau,
                                               int*                lwork)
{
    return hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnSorgqr(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    float*              A,
                                    int                 lda,
                                    float*              tau,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDorgqr(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    double*             A,
                                    int                 lda,
                                    double*             tau,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCungqr(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    tau,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZungqr(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 k,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   tau,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

// orgtr/ungtr
hipsolverStatus_t hipsolverDnSorgtr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               int*                lwork)
{
    return hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnDorgtr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               int*                lwork)
{
    return hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnCungtr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               hipFloatComplex*    tau,
                                               int*                lwork)
{
    return hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnZungtr_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               hipDoubleComplex*   tau,
                                               int*                lwork)
{
    return hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
}

hipsolverStatus_t hipsolverDnSorgtr(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              tau,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDorgtr(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             tau,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCungtr(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    tau,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZungtr(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   tau,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo);
}

// ormqr/unmqr
hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnSormqr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDormqr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCunmqr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZunmqr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

// ormtr/unmtr
hipsolverStatus_t hipsolverDnSormtr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnDormtr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnCunmtr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnZunmtr_bufferSize(hipsolverDnHandle_t  handle,
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
{
    return hipsolverZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

hipsolverStatus_t hipsolverDnSormtr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverSormtr(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDormtr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverDormtr(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCunmtr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverCunmtr(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZunmtr(hipsolverDnHandle_t  handle,
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
{
    return hipsolverZunmtr(
        handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo);
}

// gebrd
hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverSgebrd_bufferSize(handle, m, n, lwork);
}

hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverDgebrd_bufferSize(handle, m, n, lwork);
}

hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverCgebrd_bufferSize(handle, m, n, lwork);
}

hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverZgebrd_bufferSize(handle, m, n, lwork);
}

hipsolverStatus_t hipsolverDnSgebrd(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              D,
                                    float*              E,
                                    float*              tauq,
                                    float*              taup,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDgebrd(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             D,
                                    double*             E,
                                    double*             tauq,
                                    double*             taup,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCgebrd(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    float*              D,
                                    float*              E,
                                    hipFloatComplex*    tauq,
                                    hipFloatComplex*    taup,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZgebrd(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    double*             D,
                                    double*             E,
                                    hipDoubleComplex*   tauq,
                                    hipDoubleComplex*   taup,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo);
}

// gels
hipsolverStatus_t hipsolverDnSSgels_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 nrhs,
                                               float*              A,
                                               int                 lda,
                                               float*              B,
                                               int                 ldb,
                                               float*              X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnDDgels_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 nrhs,
                                               double*             A,
                                               int                 lda,
                                               double*             B,
                                               int                 ldb,
                                               double*             X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnCCgels_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 nrhs,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               hipFloatComplex*    B,
                                               int                 ldb,
                                               hipFloatComplex*    X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnZZgels_bufferSize(hipsolverDnHandle_t handle,
                                               int                 m,
                                               int                 n,
                                               int                 nrhs,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               hipDoubleComplex*   B,
                                               int                 ldb,
                                               hipDoubleComplex*   X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnSSgels(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 nrhs,
                                    float*              A,
                                    int                 lda,
                                    float*              B,
                                    int                 ldb,
                                    float*              X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverSSgels(
        handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnDDgels(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 nrhs,
                                    double*             A,
                                    int                 lda,
                                    double*             B,
                                    int                 ldb,
                                    double*             X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverDDgels(
        handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnCCgels(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 nrhs,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    B,
                                    int                 ldb,
                                    hipFloatComplex*    X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverCCgels(
        handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnZZgels(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    int                 nrhs,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   B,
                                    int                 ldb,
                                    hipDoubleComplex*   X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverZZgels(
        handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

// geqrf
hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    return hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    return hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
{
    return hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
{
    return hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnSgeqrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              tau,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDgeqrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             tau,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCgeqrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    tau,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZgeqrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   tau,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo);
}

// gesv
hipsolverStatus_t hipsolverDnSSgesv_bufferSize(hipsolverDnHandle_t handle,
                                               int                 n,
                                               int                 nrhs,
                                               float*              A,
                                               int                 lda,
                                               int*                devIpiv,
                                               float*              B,
                                               int                 ldb,
                                               float*              X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnDDgesv_bufferSize(hipsolverDnHandle_t handle,
                                               int                 n,
                                               int                 nrhs,
                                               double*             A,
                                               int                 lda,
                                               int*                devIpiv,
                                               double*             B,
                                               int                 ldb,
                                               double*             X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnCCgesv_bufferSize(hipsolverDnHandle_t handle,
                                               int                 n,
                                               int                 nrhs,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               int*                devIpiv,
                                               hipFloatComplex*    B,
                                               int                 ldb,
                                               hipFloatComplex*    X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnZZgesv_bufferSize(hipsolverDnHandle_t handle,
                                               int                 n,
                                               int                 nrhs,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               int*                devIpiv,
                                               hipDoubleComplex*   B,
                                               int                 ldb,
                                               hipDoubleComplex*   X,
                                               int                 ldx,
                                               void*               work,
                                               size_t*             lwork)
{
    return hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork);
}

hipsolverStatus_t hipsolverDnSSgesv(hipsolverDnHandle_t handle,
                                    int                 n,
                                    int                 nrhs,
                                    float*              A,
                                    int                 lda,
                                    int*                devIpiv,
                                    float*              B,
                                    int                 ldb,
                                    float*              X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverSSgesv(
        handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnDDgesv(hipsolverDnHandle_t handle,
                                    int                 n,
                                    int                 nrhs,
                                    double*             A,
                                    int                 lda,
                                    int*                devIpiv,
                                    double*             B,
                                    int                 ldb,
                                    double*             X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverDDgesv(
        handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnCCgesv(hipsolverDnHandle_t handle,
                                    int                 n,
                                    int                 nrhs,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    int*                devIpiv,
                                    hipFloatComplex*    B,
                                    int                 ldb,
                                    hipFloatComplex*    X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverCCgesv(
        handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

hipsolverStatus_t hipsolverDnZZgesv(hipsolverDnHandle_t handle,
                                    int                 n,
                                    int                 nrhs,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    int*                devIpiv,
                                    hipDoubleComplex*   B,
                                    int                 ldb,
                                    hipDoubleComplex*   X,
                                    int                 ldx,
                                    void*               work,
                                    size_t              lwork,
                                    int*                niters,
                                    int*                devInfo)
{
    return hipsolverZZgesv(
        handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, niters, devInfo);
}

// gesvd
hipsolverStatus_t hipsolverDnSgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              S,
                                    float*              U,
                                    int                 ldu,
                                    float*              V,
                                    int                 ldv,
                                    float*              work,
                                    int                 lwork,
                                    float*              rwork,
                                    int*                devInfo)
{
    return hipsolverSgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnDgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             S,
                                    double*             U,
                                    int                 ldu,
                                    double*             V,
                                    int                 ldv,
                                    double*             work,
                                    int                 lwork,
                                    double*             rwork,
                                    int*                devInfo)
{
    return hipsolverDgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnCgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    float*              S,
                                    hipFloatComplex*    U,
                                    int                 ldu,
                                    hipFloatComplex*    V,
                                    int                 ldv,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    float*              rwork,
                                    int*                devInfo)
{
    return hipsolverCgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnZgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    double*             S,
                                    hipDoubleComplex*   U,
                                    int                 ldu,
                                    hipDoubleComplex*   V,
                                    int                 ldv,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    double*             rwork,
                                    int*                devInfo)
{
    return hipsolverZgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

// getrf
hipsolverStatus_t hipsolverDnSgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnDgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
{
    return hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
{
    return hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

// getrs
hipsolverStatus_t hipsolverDnSgetrs(hipsolverDnHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                  n,
                                    int                  nrhs,
                                    float*               A,
                                    int                  lda,
                                    int*                 devIpiv,
                                    float*               B,
                                    int                  ldb,
                                    int*                 devInfo)
{
    return hipsolverSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnDgetrs(hipsolverDnHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                  n,
                                    int                  nrhs,
                                    double*              A,
                                    int                  lda,
                                    int*                 devIpiv,
                                    double*              B,
                                    int                  ldb,
                                    int*                 devInfo)
{
    return hipsolverDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnCgetrs(hipsolverDnHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                  n,
                                    int                  nrhs,
                                    hipFloatComplex*     A,
                                    int                  lda,
                                    int*                 devIpiv,
                                    hipFloatComplex*     B,
                                    int                  ldb,
                                    int*                 devInfo)
{
    return hipsolverCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnZgetrs(hipsolverDnHandle_t  handle,
                                    hipsolverOperation_t trans,
                                    int                  n,
                                    int                  nrhs,
                                    hipDoubleComplex*    A,
                                    int                  lda,
                                    int*                 devIpiv,
                                    hipDoubleComplex*    B,
                                    int                  ldb,
                                    int*                 devInfo)
{
    return hipsolverZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, nullptr, 0, devInfo);
}

// potrf
hipsolverStatus_t hipsolverDnSpotrf_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
{
    return hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnDpotrf_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
{
    return hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               int*                lwork)
{
    return hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               int*                lwork)
{
    return hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnSpotrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDpotrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCpotrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZpotrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, devInfo);
}

// potrf_batched
hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           float*              A[],
                                           int                 lda,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverSpotrfBatched(handle, uplo, n, A, lda, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           double*             A[],
                                           int                 lda,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverDpotrfBatched(handle, uplo, n, A, lda, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           hipFloatComplex*    A[],
                                           int                 lda,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverCpotrfBatched(handle, uplo, n, A, lda, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           hipDoubleComplex*   A[],
                                           int                 lda,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverZpotrfBatched(handle, uplo, n, A, lda, nullptr, 0, devInfo, batch_count);
}

// potri
hipsolverStatus_t hipsolverDnSpotri_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork)
{
    return hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnDpotri_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork)
{
    return hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCpotri_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               int*                lwork)
{
    return hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZpotri_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               int*                lwork)
{
    return hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnSpotri(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDpotri(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCpotri(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCpotri(handle, uplo, n, A, lda, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZpotri(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZpotri(handle, uplo, n, A, lda, work, lwork, devInfo);
}

// potrs
hipsolverStatus_t hipsolverDnSpotrs(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    int                 nrhs,
                                    float*              A,
                                    int                 lda,
                                    float*              B,
                                    int                 ldb,
                                    int*                devInfo)
{
    return hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnDpotrs(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    int                 nrhs,
                                    double*             A,
                                    int                 lda,
                                    double*             B,
                                    int                 ldb,
                                    int*                devInfo)
{
    return hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnCpotrs(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    int                 nrhs,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    B,
                                    int                 ldb,
                                    int*                devInfo)
{
    return hipsolverCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo);
}

hipsolverStatus_t hipsolverDnZpotrs(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    int                 nrhs,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   B,
                                    int                 ldb,
                                    int*                devInfo)
{
    return hipsolverZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo);
}

// potrs_batched
hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           int                 nrhs,
                                           float*              A[],
                                           int                 lda,
                                           float*              B[],
                                           int                 ldb,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverSpotrsBatched(
        handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           int                 nrhs,
                                           double*             A[],
                                           int                 lda,
                                           double*             B[],
                                           int                 ldb,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverDpotrsBatched(
        handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           int                 nrhs,
                                           hipFloatComplex*    A[],
                                           int                 lda,
                                           hipFloatComplex*    B[],
                                           int                 ldb,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverCpotrsBatched(
        handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo, batch_count);
}

hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverDnHandle_t handle,
                                           hipsolverFillMode_t uplo,
                                           int                 n,
                                           int                 nrhs,
                                           hipDoubleComplex*   A[],
                                           int                 lda,
                                           hipDoubleComplex*   B[],
                                           int                 ldb,
                                           int*                devInfo,
                                           int                 batch_count)
{
    return hipsolverZpotrsBatched(
        handle, uplo, n, nrhs, A, lda, B, ldb, nullptr, 0, devInfo, batch_count);
}

// syevd/heevd
hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              D,
                                               int*                lwork)
{
    return hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
}

hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             D,
                                               int*                lwork)
{
    return hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
}

hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               float*              D,
                                               int*                lwork)
{
    return hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
}

hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               double*             D,
                                               int*                lwork)
{
    return hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
}

hipsolverStatus_t hipsolverDnSsyevd(hipsolverDnHandle_t handle,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              D,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDsyevd(hipsolverDnHandle_t handle,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             D,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCheevd(hipsolverDnHandle_t handle,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    float*              D,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZheevd(hipsolverDnHandle_t handle,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    double*             D,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo);
}

// sygvd/hegvd
hipsolverStatus_t hipsolverDnSsygvd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              B,
                                               int                 ldb,
                                               float*              D,
                                               int*                lwork)
{
    return hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

hipsolverStatus_t hipsolverDnDsygvd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             B,
                                               int                 ldb,
                                               double*             D,
                                               int*                lwork)
{
    return hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

hipsolverStatus_t hipsolverDnChegvd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               hipFloatComplex*    B,
                                               int                 ldb,
                                               float*              D,
                                               int*                lwork)
{
    return hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

hipsolverStatus_t hipsolverDnZhegvd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               hipDoubleComplex*   B,
                                               int                 ldb,
                                               double*             D,
                                               int*                lwork)
{
    return hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

hipsolverStatus_t hipsolverDnSsygvd(hipsolverDnHandle_t handle,
                                    hipsolverEigType_t  itype,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              B,
                                    int                 ldb,
                                    float*              D,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDsygvd(hipsolverDnHandle_t handle,
                                    hipsolverEigType_t  itype,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             B,
                                    int                 ldb,
                                    double*             D,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnChegvd(hipsolverDnHandle_t handle,
                                    hipsolverEigType_t  itype,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    B,
                                    int                 ldb,
                                    float*              D,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZhegvd(hipsolverDnHandle_t handle,
                                    hipsolverEigType_t  itype,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   B,
                                    int                 ldb,
                                    double*             D,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, devInfo);
}

// sytrd/hetrd
hipsolverStatus_t hipsolverDnSsytrd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              D,
                                               float*              E,
                                               float*              tau,
                                               int*                lwork)
{
    return hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
}

hipsolverStatus_t hipsolverDnDsytrd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             D,
                                               double*             E,
                                               double*             tau,
                                               int*                lwork)
{
    return hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
}

hipsolverStatus_t hipsolverDnChetrd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipFloatComplex*    A,
                                               int                 lda,
                                               float*              D,
                                               float*              E,
                                               hipFloatComplex*    tau,
                                               int*                lwork)
{
    return hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
}

hipsolverStatus_t hipsolverDnZhetrd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipDoubleComplex*   A,
                                               int                 lda,
                                               double*             D,
                                               double*             E,
                                               hipDoubleComplex*   tau,
                                               int*                lwork)
{
    return hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
}

hipsolverStatus_t hipsolverDnSsytrd(hipsolverDnHandle_t handle,
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
{
    return hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDsytrd(hipsolverDnHandle_t handle,
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
{
    return hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnChetrd(hipsolverDnHandle_t handle,
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
{
    return hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZhetrd(hipsolverDnHandle_t handle,
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
{
    return hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo);
}

// sytrf
hipsolverStatus_t
    hipsolverDnSsytrf_bufferSize(hipsolverDnHandle_t handle, int n, float* A, int lda, int* lwork)
{
    return hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork);
}

hipsolverStatus_t
    hipsolverDnDsytrf_bufferSize(hipsolverDnHandle_t handle, int n, double* A, int lda, int* lwork)
{
    return hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCsytrf_bufferSize(
    hipsolverDnHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork)
{
    return hipsolverCsytrf_bufferSize(handle, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZsytrf_bufferSize(
    hipsolverDnHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork)
{
    return hipsolverZsytrf_bufferSize(handle, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnSsytrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    int*                ipiv,
                                    float*              work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnDsytrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    int*                ipiv,
                                    double*             work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnCsytrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    int*                ipiv,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
}

hipsolverStatus_t hipsolverDnZsytrf(hipsolverDnHandle_t handle,
                                    hipsolverFillMode_t uplo,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    int*                ipiv,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    int*                devInfo)
{
    return hipsolverZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo);
}

} //extern C
