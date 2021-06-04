/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"

/*!\file
 *  This file interfaces with our Fortran LAPACK interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */

extern "C" {

/* ==========
 *   LAPACK
 * ========== */

// orgbr/ungbr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     hipsolverComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbr_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverSideMode_t     side,
                                                                     int                     m,
                                                                     int                     n,
                                                                     int                     k,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     hipsolverDoubleComplex* tau,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbrFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbrFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbrFortran(hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbrFortran(hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// orgqr/ungqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqr_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqr_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqr_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               k,
                                                                     hipsolverComplex* A,
                                                                     int               lda,
                                                                     hipsolverComplex* tau,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqr_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     int                     m,
                                                                     int                     n,
                                                                     int                     k,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     hipsolverDoubleComplex* tau,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* tau,
                                                          hipsolverComplex* work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqrFortran(hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// orgtr/ungtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     hipsolverComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtr_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverFillMode_t     uplo,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     hipsolverDoubleComplex* tau,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtrFortran(hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// ormqr/unmqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqr_bufferSizeFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqr_bufferSizeFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqr_bufferSizeFortran(hipsolverHandle_t    handle,
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
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqr_bufferSizeFortran(hipsolverHandle_t       handle,
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
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqrFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqrFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqrFortran(hipsolverHandle_t    handle,
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
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqrFortran(hipsolverHandle_t       handle,
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
                                                          int*                    devInfo);

// ormtr/unmtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtr_bufferSizeFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtr_bufferSizeFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverFillMode_t  uplo,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     hipsolverComplex*    A,
                                                                     int                  lda,
                                                                     hipsolverComplex*    tau,
                                                                     hipsolverComplex*    C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtr_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverSideMode_t     side,
                                                                     hipsolverFillMode_t     uplo,
                                                                     hipsolverOperation_t    trans,
                                                                     int                     m,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     hipsolverDoubleComplex* tau,
                                                                     hipsolverDoubleComplex* C,
                                                                     int                     ldc,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtrFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtrFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    tau,
                                                          hipsolverComplex*    C,
                                                          int                  ldc,
                                                          hipsolverComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtrFortran(hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          hipsolverFillMode_t     uplo,
                                                          hipsolverOperation_t    trans,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* C,
                                                          int                     ldc,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// gebrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrdFortran(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrdFortran(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrdFortran(hipsolverHandle_t handle,
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
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrdFortran(hipsolverHandle_t       handle,
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
                                                          int*                    devInfo);

// geqrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* tau,
                                                          hipsolverComplex* work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrfFortran(hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// gesvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvdFortran(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvdFortran(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvdFortran(hipsolverHandle_t handle,
                                                          signed char       jobu,
                                                          signed char       jobv,
                                                          int               m,
                                                          int               n,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          float*            S,
                                                          hipsolverComplex* U,
                                                          int               ldu,
                                                          hipsolverComplex* V,
                                                          int               ldv,
                                                          hipsolverComplex* work,
                                                          int               lwork,
                                                          float*            rwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvdFortran(hipsolverHandle_t       handle,
                                                          signed char             jobu,
                                                          signed char             jobv,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 S,
                                                          hipsolverDoubleComplex* U,
                                                          int                     ldu,
                                                          hipsolverDoubleComplex* V,
                                                          int                     ldv,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          double*                 rwork,
                                                          int*                    devInfo);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrfFortran(hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devIpiv,
                                                          int*                    devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     float*               B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     double*              B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     hipsolverComplex*    A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     hipsolverComplex*    B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrs_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverOperation_t    trans,
                                                                     int                     n,
                                                                     int                     nrhs,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     int* devIpiv,
                                                                     hipsolverDoubleComplex* B,
                                                                     int                     ldb,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrsFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrsFortran(hipsolverHandle_t    handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          hipsolverComplex*    B,
                                                          int                  ldb,
                                                          hipsolverComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrsFortran(hipsolverHandle_t       handle,
                                                          hipsolverOperation_t    trans,
                                                          int                     n,
                                                          int                     nrhs,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          int*                    devIpiv,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// potrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverFillMode_t     uplo,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfFortran(hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// potrf_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverSpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverCpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipsolverComplex*   A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverZpotrfBatched_bufferSizeFortran(hipsolverHandle_t       handle,
                                             hipsolverFillMode_t     uplo,
                                             int                     n,
                                             hipsolverDoubleComplex* A[],
                                             int                     lda,
                                             int*                    lwork,
                                             int                     batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 float*              A[],
                                                                 int                 lda,
                                                                 float*              work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 double*             A[],
                                                                 int                 lda,
                                                                 double*             work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 hipsolverComplex*   A[],
                                                                 int                 lda,
                                                                 hipsolverComplex*   work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfBatchedFortran(hipsolverHandle_t       handle,
                                                                 hipsolverFillMode_t     uplo,
                                                                 int                     n,
                                                                 hipsolverDoubleComplex* A[],
                                                                 int                     lda,
                                                                 hipsolverDoubleComplex* work,
                                                                 int                     lwork,
                                                                 int*                    devInfo,
                                                                 int batch_count);

// syevd/heevd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             D,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevd_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverEigMode_t      jobz,
                                                                     hipsolverFillMode_t     uplo,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     double*                 D,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevdFortran(hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// sygvd/hegvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd_bufferSizeFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd_bufferSizeFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigType_t  itype,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     hipsolverComplex*   B,
                                                                     int                 ldb,
                                                                     float*              D,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverEigType_t      itype,
                                                                     hipsolverEigMode_t      jobz,
                                                                     hipsolverFillMode_t     uplo,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     hipsolverDoubleComplex* B,
                                                                     int                     ldb,
                                                                     double*                 D,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvdFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvdFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   B,
                                                          int                 ldb,
                                                          float*              D,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvdFortran(hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 D,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);

// sytrd/hetrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     float*              E,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             D,
                                                                     double*             E,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipsolverComplex*   A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     float*              E,
                                                                     hipsolverComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrd_bufferSizeFortran(hipsolverHandle_t       handle,
                                                                     hipsolverFillMode_t     uplo,
                                                                     int                     n,
                                                                     hipsolverDoubleComplex* A,
                                                                     int                     lda,
                                                                     double*                 D,
                                                                     double*                 E,
                                                                     hipsolverDoubleComplex* tau,
                                                                     int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrdFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrdFortran(hipsolverHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrdFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          hipsolverComplex*   tau,
                                                          hipsolverComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrdFortran(hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          double*                 E,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* work,
                                                          int                     lwork,
                                                          int*                    devInfo);
}
