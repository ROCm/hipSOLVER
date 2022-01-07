/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-compat.h provides a compatibility API for users of
 *  cuSOLVER wishing to transition to hipSOLVER. Functions in this file
 *  have parameter lists matching those of cuSOLVER, but may suffer from
 *  performance issues when using the rocSOLVER backend. Switching to the
 *  use of functions from hipsolver-functions.h is recommended.
 */

#ifndef HIPSOLVER_COMPAT_H
#define HIPSOLVER_COMPAT_H

#include "hipsolver-functions.h"
#include "hipsolver-types.h"

typedef void* hipsolverDnHandle_t;

typedef void* hipsolverSyevjInfo_t;

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreate(hipsolverDnHandle_t* handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroy(hipsolverDnHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSetStream(hipsolverDnHandle_t handle,
                                                        hipStream_t         streamId);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnGetStream(hipsolverDnHandle_t handle,
                                                        hipStream_t*        streamId);

// params
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateSyevjInfo(hipsolverSyevjInfo_t* info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroySyevjInfo(hipsolverSyevjInfo_t info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info,
                                                                 int                  max_sweeps);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetSortEig(hipsolverSyevjInfo_t info,
                                                               int                  sort_eig);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetTolerance(hipsolverSyevjInfo_t info,
                                                                 double               tolerance);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetResidual(hipsolverDnHandle_t  handle,
                                                                hipsolverSyevjInfo_t info,
                                                                double*              residual);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetSweeps(hipsolverDnHandle_t  handle,
                                                              hipsolverSyevjInfo_t info,
                                                              int*                 executed_sweeps);

// orgbr/ungbr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverSideMode_t side,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                float*              A,
                                                                int                 lda,
                                                                float*              tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverSideMode_t side,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                double*             A,
                                                                int                 lda,
                                                                double*             tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverSideMode_t side,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                hipFloatComplex*    tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverSideMode_t side,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                hipDoubleComplex*   tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr(hipsolverDnHandle_t handle,
                                                     hipsolverSideMode_t side,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              tau,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr(hipsolverDnHandle_t handle,
                                                     hipsolverSideMode_t side,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             tau,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr(hipsolverDnHandle_t handle,
                                                     hipsolverSideMode_t side,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    tau,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr(hipsolverDnHandle_t handle,
                                                     hipsolverSideMode_t side,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   tau,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// orgqr/ungqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                hipFloatComplex*    tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int                 k,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                hipDoubleComplex*   tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              tau,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             tau,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    tau,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     int                 k,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   tau,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// orgtr/ungtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                float*              A,
                                                                int                 lda,
                                                                float*              tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                double*             A,
                                                                int                 lda,
                                                                double*             tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                hipFloatComplex*    tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                hipDoubleComplex*   tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              tau,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             tau,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    tau,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   tau,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// ormqr/unmqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

// ormtr/unmtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr_bufferSize(hipsolverDnHandle_t  handle,
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
                                                                int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr(hipsolverDnHandle_t  handle,
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
                                                     int*                 devInfo);

// gebrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

// gels
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

// geqrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              tau,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             tau,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    tau,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   tau,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// gesv
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv_bufferSize(hipsolverDnHandle_t handle,
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
                                                                size_t*             lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

// gesvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverDnHandle_t handle,
                                                                int                 m,
                                                                int                 n,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              work,
                                                     int*                devIpiv,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             work,
                                                     int*                devIpiv,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    work,
                                                     int*                devIpiv,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf(hipsolverDnHandle_t handle,
                                                     int                 m,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   work,
                                                     int*                devIpiv,
                                                     int*                devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverOperation_t trans,
                                                     int                  n,
                                                     int                  nrhs,
                                                     float*               A,
                                                     int                  lda,
                                                     int*                 devIpiv,
                                                     float*               B,
                                                     int                  ldb,
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverOperation_t trans,
                                                     int                  n,
                                                     int                  nrhs,
                                                     double*              A,
                                                     int                  lda,
                                                     int*                 devIpiv,
                                                     double*              B,
                                                     int                  ldb,
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverOperation_t trans,
                                                     int                  n,
                                                     int                  nrhs,
                                                     hipFloatComplex*     A,
                                                     int                  lda,
                                                     int*                 devIpiv,
                                                     hipFloatComplex*     B,
                                                     int                  ldb,
                                                     int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverOperation_t trans,
                                                     int                  n,
                                                     int                  nrhs,
                                                     hipDoubleComplex*    A,
                                                     int                  lda,
                                                     int*                 devIpiv,
                                                     hipDoubleComplex*    B,
                                                     int                  ldb,
                                                     int*                 devInfo);

// potrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// potrf_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            float*              A[],
                                                            int                 lda,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            double*             A[],
                                                            int                 lda,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipFloatComplex*    A[],
                                                            int                 lda,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipDoubleComplex*   A[],
                                                            int                 lda,
                                                            int*                devInfo,
                                                            int                 batch_count);

// potri
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri_bufferSize(
    hipsolverDnHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// potrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrs(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     int                 nrhs,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              B,
                                                     int                 ldb,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrs(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     int                 nrhs,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             B,
                                                     int                 ldb,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrs(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     int                 nrhs,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     hipFloatComplex*    B,
                                                     int                 ldb,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrs(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     int                 nrhs,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     hipDoubleComplex*   B,
                                                     int                 ldb,
                                                     int*                devInfo);

// potrs_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            int                 nrhs,
                                                            float*              A[],
                                                            int                 lda,
                                                            float*              B[],
                                                            int                 ldb,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            int                 nrhs,
                                                            double*             A[],
                                                            int                 lda,
                                                            double*             B[],
                                                            int                 ldb,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            int                 nrhs,
                                                            hipFloatComplex*    A[],
                                                            int                 lda,
                                                            hipFloatComplex*    B[],
                                                            int                 ldb,
                                                            int*                devInfo,
                                                            int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverDnHandle_t handle,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            int                 nrhs,
                                                            hipDoubleComplex*   A[],
                                                            int                 lda,
                                                            hipDoubleComplex*   B[],
                                                            int                 ldb,
                                                            int*                devInfo,
                                                            int                 batch_count);

// syevd/heevd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                float*              A,
                                                                int                 lda,
                                                                float*              D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                double*             A,
                                                                int                 lda,
                                                                double*             D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                float*              D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                double*             D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd(hipsolverDnHandle_t handle,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              D,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd(hipsolverDnHandle_t handle,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             D,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd(hipsolverDnHandle_t handle,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     float*              D,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd(hipsolverDnHandle_t handle,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     double*             D,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// syevj/heevj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipsolverFillMode_t  uplo,
                                                                int                  n,
                                                                float*               A,
                                                                int                  lda,
                                                                float*               D,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipsolverFillMode_t  uplo,
                                                                int                  n,
                                                                double*              A,
                                                                int                  lda,
                                                                double*              D,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipsolverFillMode_t  uplo,
                                                                int                  n,
                                                                hipFloatComplex*     A,
                                                                int                  lda,
                                                                float*               D,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipsolverFillMode_t  uplo,
                                                                int                  n,
                                                                hipDoubleComplex*    A,
                                                                int                  lda,
                                                                double*              D,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipsolverFillMode_t  uplo,
                                                     int                  n,
                                                     float*               A,
                                                     int                  lda,
                                                     float*               D,
                                                     float*               work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipsolverFillMode_t  uplo,
                                                     int                  n,
                                                     double*              A,
                                                     int                  lda,
                                                     double*              D,
                                                     double*              work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipsolverFillMode_t  uplo,
                                                     int                  n,
                                                     hipFloatComplex*     A,
                                                     int                  lda,
                                                     float*               D,
                                                     hipFloatComplex*     work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipsolverFillMode_t  uplo,
                                                     int                  n,
                                                     hipDoubleComplex*    A,
                                                     int                  lda,
                                                     double*              D,
                                                     hipDoubleComplex*    work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

// syevj_batched/heevj_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipsolverFillMode_t  uplo,
                                                                       int                  n,
                                                                       float*               A,
                                                                       int                  lda,
                                                                       float*               D,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipsolverFillMode_t  uplo,
                                                                       int                  n,
                                                                       double*              A,
                                                                       int                  lda,
                                                                       double*              D,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipsolverFillMode_t  uplo,
                                                                       int                  n,
                                                                       hipFloatComplex*     A,
                                                                       int                  lda,
                                                                       float*               D,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipsolverFillMode_t  uplo,
                                                                       int                  n,
                                                                       hipDoubleComplex*    A,
                                                                       int                  lda,
                                                                       double*              D,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipsolverFillMode_t  uplo,
                                                            int                  n,
                                                            float*               A,
                                                            int                  lda,
                                                            float*               D,
                                                            float*               work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipsolverFillMode_t  uplo,
                                                            int                  n,
                                                            double*              A,
                                                            int                  lda,
                                                            double*              D,
                                                            double*              work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipsolverFillMode_t  uplo,
                                                            int                  n,
                                                            hipFloatComplex*     A,
                                                            int                  lda,
                                                            float*               D,
                                                            hipFloatComplex*     work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipsolverFillMode_t  uplo,
                                                            int                  n,
                                                            hipDoubleComplex*    A,
                                                            int                  lda,
                                                            double*              D,
                                                            hipDoubleComplex*    work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

// sygvd/hegvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigType_t  itype,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                float*              A,
                                                                int                 lda,
                                                                float*              B,
                                                                int                 ldb,
                                                                float*              D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigType_t  itype,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                double*             A,
                                                                int                 lda,
                                                                double*             B,
                                                                int                 ldb,
                                                                double*             D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigType_t  itype,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                hipFloatComplex*    B,
                                                                int                 ldb,
                                                                float*              D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverEigType_t  itype,
                                                                hipsolverEigMode_t  jobz,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                hipDoubleComplex*   B,
                                                                int                 ldb,
                                                                double*             D,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd(hipsolverDnHandle_t handle,
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
                                                     int*                devInfo);

// sytrd/hetrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                float*              A,
                                                                int                 lda,
                                                                float*              D,
                                                                float*              E,
                                                                float*              tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                double*             A,
                                                                int                 lda,
                                                                double*             D,
                                                                double*             E,
                                                                double*             tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipFloatComplex*    A,
                                                                int                 lda,
                                                                float*              D,
                                                                float*              E,
                                                                hipFloatComplex*    tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd_bufferSize(hipsolverDnHandle_t handle,
                                                                hipsolverFillMode_t uplo,
                                                                int                 n,
                                                                hipDoubleComplex*   A,
                                                                int                 lda,
                                                                double*             D,
                                                                double*             E,
                                                                hipDoubleComplex*   tau,
                                                                int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     float*              D,
                                                     float*              E,
                                                     float*              tau,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     double*             D,
                                                     double*             E,
                                                     double*             tau,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     float*              D,
                                                     float*              E,
                                                     hipFloatComplex*    tau,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     double*             D,
                                                     double*             E,
                                                     hipDoubleComplex*   tau,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

// sytrf
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnSsytrf_bufferSize(hipsolverDnHandle_t handle, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnDsytrf_bufferSize(hipsolverDnHandle_t handle, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf_bufferSize(
    hipsolverDnHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf_bufferSize(
    hipsolverDnHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     float*              A,
                                                     int                 lda,
                                                     int*                ipiv,
                                                     float*              work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     double*             A,
                                                     int                 lda,
                                                     int*                ipiv,
                                                     double*             work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipFloatComplex*    A,
                                                     int                 lda,
                                                     int*                ipiv,
                                                     hipFloatComplex*    work,
                                                     int                 lwork,
                                                     int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf(hipsolverDnHandle_t handle,
                                                     hipsolverFillMode_t uplo,
                                                     int                 n,
                                                     hipDoubleComplex*   A,
                                                     int                 lda,
                                                     int*                ipiv,
                                                     hipDoubleComplex*   work,
                                                     int                 lwork,
                                                     int*                devInfo);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_COMPAT_H
