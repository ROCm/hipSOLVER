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
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           work,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* work,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrfFortran(hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* work,
                                                          int*                    devIpiv,
                                                          int*                    devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          float*               A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          float*               B,
                                                          int                  ldb,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 float*              A[],
                                                                 int                 lda,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 double*             A[],
                                                                 int                 lda,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 hipsolverComplex*   A[],
                                                                 int                 lda,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfBatchedFortran(hipsolverHandle_t       handle,
                                                                 hipsolverFillMode_t     uplo,
                                                                 int                     n,
                                                                 hipsolverDoubleComplex* A[],
                                                                 int                     lda,
                                                                 int*                    devInfo,
                                                                 int batch_count);

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
