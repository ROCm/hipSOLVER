/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-functions.h provides function wrappers around cuSOLVER
 *  and rocSOLVER functions. Some differences with the cuSOLVER API are
 *  present in order to accomodate the needs of rocSOLVER.
 */

#ifndef HIPSOLVER_FUNCTIONS_H
#define HIPSOLVER_FUNCTIONS_H

#include "hipsolver-types.h"
#include <hip/hip_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle,
                                                      hipStream_t       streamId);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle,
                                                      hipStream_t*      streamId);

// orgbr/ungbr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverSideMode_t side,
                                                              int                 m,
                                                              int                 n,
                                                              int                 k,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbr(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbr(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbr(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbr(hipsolverHandle_t   handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqr_bufferSize(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqr_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               k,
                                                              hipFloatComplex*  A,
                                                              int               lda,
                                                              hipFloatComplex*  tau,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqr_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               k,
                                                              hipDoubleComplex* A,
                                                              int               lda,
                                                              hipDoubleComplex* tau,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   float*            A,
                                                   int               lda,
                                                   float*            tau,
                                                   float*            work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   double*           A,
                                                   int               lda,
                                                   double*           tau,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  tau,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqr(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               k,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* tau,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devInfo);

// orgtr/ungtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtr_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtr(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              tau,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtr(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             tau,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtr(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    tau,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtr(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   tau,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

// ormqr/unmqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqr(hipsolverHandle_t    handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtr_bufferSize(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtr(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtr(hipsolverHandle_t    handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrd_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrd_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrd_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrd_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

// gels
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              float*            A,
                                                              int               lda,
                                                              float*            B,
                                                              int               ldb,
                                                              float*            X,
                                                              int               ldx,
                                                              size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              double*           A,
                                                              int               lda,
                                                              double*           B,
                                                              int               ldb,
                                                              double*           X,
                                                              int               ldx,
                                                              size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              hipFloatComplex*  A,
                                                              int               lda,
                                                              hipFloatComplex*  B,
                                                              int               ldb,
                                                              hipFloatComplex*  X,
                                                              int               ldx,
                                                              size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgels_bufferSize(hipsolverHandle_t handle,
                                                              int               m,
                                                              int               n,
                                                              int               nrhs,
                                                              hipDoubleComplex* A,
                                                              int               lda,
                                                              hipDoubleComplex* B,
                                                              int               ldb,
                                                              hipDoubleComplex* X,
                                                              int               ldx,
                                                              size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgels(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgels(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgels(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgels(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

// geqrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   float*            A,
                                                   int               lda,
                                                   float*            tau,
                                                   float*            work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           tau,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  tau,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* tau,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devInfo);

// gesv
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
                                                              size_t*           lwork);

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
                                                              size_t*           lwork);

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
                                                              size_t*           lwork);

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
                                                              size_t*           lwork);

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
                                                   int*              devInfo);

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
                                                   int*              devInfo);

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
                                                   int*              devInfo);

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
                                                   int*              devInfo);

// gesvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvd_bufferSize(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvd(hipsolverHandle_t handle,
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
                                                   int*              devInfo);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   float*            A,
                                                   int               lda,
                                                   float*            work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipFloatComplex*  A,
                                                   int               lda,
                                                   hipFloatComplex*  work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipDoubleComplex* A,
                                                   int               lda,
                                                   hipDoubleComplex* work,
                                                   int               lwork,
                                                   int*              devIpiv,
                                                   int*              devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              float*               A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              float*               B,
                                                              int                  ldb,
                                                              int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              double*              A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              double*              B,
                                                              int                  ldb,
                                                              int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              hipFloatComplex*     A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              hipFloatComplex*     B,
                                                              int                  ldb,
                                                              int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrs_bufferSize(hipsolverHandle_t    handle,
                                                              hipsolverOperation_t trans,
                                                              int                  n,
                                                              int                  nrhs,
                                                              hipDoubleComplex*    A,
                                                              int                  lda,
                                                              int*                 devIpiv,
                                                              hipDoubleComplex*    B,
                                                              int                  ldb,
                                                              int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
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
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
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
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
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
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t    handle,
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
                                                   int*                 devInfo);

// potrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

// potrf_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A[],
                                                                     int                 lda,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A[],
                                                          int                 lda,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A[],
                                                          int                 lda,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A[],
                                                          int                 lda,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfBatched(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A[],
                                                          int                 lda,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo,
                                                          int                 batch_count);

// potri
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotri_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotri_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotri_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotri(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

// potrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              B,
                                                              int                 ldb,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             B,
                                                              int                 ldb,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              hipFloatComplex*    B,
                                                              int                 ldb,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrs_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              int                 nrhs,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              hipDoubleComplex*   B,
                                                              int                 ldb,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              B,
                                                   int                 ldb,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             B,
                                                   int                 ldb,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   hipFloatComplex*    B,
                                                   int                 ldb,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrs(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   int                 nrhs,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   hipDoubleComplex*   B,
                                                   int                 ldb,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

// potrs_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     float*              A[],
                                                                     int                 lda,
                                                                     float*              B[],
                                                                     int                 ldb,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     double*             A[],
                                                                     int                 lda,
                                                                     double*             B[],
                                                                     int                 ldb,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     hipFloatComplex*    A[],
                                                                     int                 lda,
                                                                     hipFloatComplex*    B[],
                                                                     int                 ldb,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrsBatched_bufferSize(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     hipDoubleComplex*   A[],
                                                                     int                 lda,
                                                                     hipDoubleComplex*   B[],
                                                                     int                 ldb,
                                                                     int*                lwork,
                                                                     int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrsBatched(hipsolverHandle_t   handle,
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
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrsBatched(hipsolverHandle_t   handle,
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
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrsBatched(hipsolverHandle_t   handle,
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
                                                          int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrsBatched(hipsolverHandle_t   handle,
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
                                                          int                 batch_count);

// syevd/heevd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              D,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             D,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              float*              D,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverEigMode_t  jobz,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              double*             D,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevd(hipsolverHandle_t   handle,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              D,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevd(hipsolverHandle_t   handle,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             D,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevd(hipsolverHandle_t   handle,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   float*              D,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevd(hipsolverHandle_t   handle,
                                                   hipsolverEigMode_t  jobz,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipDoubleComplex*   A,
                                                   int                 lda,
                                                   double*             D,
                                                   hipDoubleComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

// sygvd/hegvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd_bufferSize(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd_bufferSize(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd_bufferSize(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd_bufferSize(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd(hipsolverHandle_t   handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              float*              A,
                                                              int                 lda,
                                                              float*              D,
                                                              float*              E,
                                                              float*              tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              double*             A,
                                                              int                 lda,
                                                              double*             D,
                                                              double*             E,
                                                              double*             tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipFloatComplex*    A,
                                                              int                 lda,
                                                              float*              D,
                                                              float*              E,
                                                              hipFloatComplex*    tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrd_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipDoubleComplex*   A,
                                                              int                 lda,
                                                              double*             D,
                                                              double*             E,
                                                              hipDoubleComplex*   tau,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrd(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrd(hipsolverHandle_t   handle,
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
    hipsolverSsytrf_bufferSize(hipsolverHandle_t handle, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   int*                ipiv,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   int*                ipiv,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCsytrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipFloatComplex*    A,
                                                   int                 lda,
                                                   int*                ipiv,
                                                   hipFloatComplex*    work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZsytrf(hipsolverHandle_t   handle,
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

#endif // HIPSOLVER_FUNCTIONS_H
