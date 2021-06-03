/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include "hipsolver_fortran.hpp"

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

typedef enum
{
    C_NORMAL,
    C_NORMAL_ALT,
    FORTRAN_NORMAL,
    FORTRAN_NORMAL_ALT,
    C_STRIDED,
    C_STRIDED_ALT,
    FORTRAN_STRIDED,
    FORTRAN_STRIDED_ALT
} testMarshal_t;

inline testMarshal_t bool2marshal(bool FORTRAN, bool ALT)
{
    if(!FORTRAN)
        if(!ALT)
            return C_NORMAL;
        else
            return C_NORMAL_ALT;
    else if(!ALT)
        return FORTRAN_NORMAL;
    else
        return FORTRAN_NORMAL_ALT;
}

/******************** ORGBR/UNGBR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverSorgbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverDorgbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverCungbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverZungbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverSorgbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverDorgbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverCungbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    if(!FORTRAN)
        return hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverZungbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
}

/******************** ORGQR/UNGQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          int*              lwork)
{
    if(!FORTRAN)
        return hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverSorgqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          int*              lwork)
{
    if(!FORTRAN)
        return hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverDorgqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool              FORTRAN,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* tau,
                                                          int*              lwork)
{
    if(!FORTRAN)
        return hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverCungqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    else
        return hipsolverZungqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               float*            A,
                                               int               lda,
                                               float*            tau,
                                               float*            work,
                                               int               lwork,
                                               int*              info)
{
    if(!FORTRAN)
        return hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverSorgqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               double*           A,
                                               int               lda,
                                               double*           tau,
                                               double*           work,
                                               int               lwork,
                                               int*              info)
{
    if(!FORTRAN)
        return hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverDorgqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool              FORTRAN,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               hipsolverComplex* A,
                                               int               lda,
                                               hipsolverComplex* tau,
                                               hipsolverComplex* work,
                                               int               lwork,
                                               int*              info)
{
    if(!FORTRAN)
        return hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverCungqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    if(!FORTRAN)
        return hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    else
        return hipsolverZungqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
}
/********************************************************/

/******************** ORGTR/UNGTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    else
        return hipsolverSorgtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    else
        return hipsolverDorgtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    else
        return hipsolverCungtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    else
        return hipsolverZungtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverSorgtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverDorgtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{
    if(!FORTRAN)
        return hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverCungtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    if(!FORTRAN)
        return hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverZungtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
}
/********************************************************/

/******************** ORMQR/UNMQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
    if(!FORTRAN)
        return hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverSormqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
    if(!FORTRAN)
        return hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverDormqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
{
    if(!FORTRAN)
        return hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverCunmqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
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
{
    if(!FORTRAN)
        return hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverZunmqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverSormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverSormqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverDormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverDormqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverCunmqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverCunmqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
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
                                               int*                    info)
{
    if(!FORTRAN)
        return hipsolverZunmqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverZunmqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
}
/********************************************************/

/******************** GEBRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverSgebrd_bufferSize(handle, m, n, lwork);
    else
        return hipsolverSgebrd_bufferSizeFortran(handle, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverDgebrd_bufferSize(handle, m, n, lwork);
    else
        return hipsolverDgebrd_bufferSizeFortran(handle, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverCgebrd_bufferSize(handle, m, n, lwork);
    else
        return hipsolverCgebrd_bufferSizeFortran(handle, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZgebrd_bufferSize(handle, m, n, lwork);
    else
        return hipsolverZgebrd_bufferSizeFortran(handle, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         float*            tauq,
                                         int               stQ,
                                         float*            taup,
                                         int               stP,
                                         float*            work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    else
        return hipsolverSgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           D,
                                         int               stD,
                                         double*           E,
                                         int               stE,
                                         double*           tauq,
                                         int               stQ,
                                         double*           taup,
                                         int               stP,
                                         double*           work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    else
        return hipsolverDgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_gebrd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         hipsolverComplex* tauq,
                                         int               stQ,
                                         hipsolverComplex* taup,
                                         int               stP,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    else
        return hipsolverCgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_gebrd(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 D,
                                         int                     stD,
                                         double*                 E,
                                         int                     stE,
                                         hipsolverDoubleComplex* tauq,
                                         int                     stQ,
                                         hipsolverDoubleComplex* taup,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    else
        return hipsolverZgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverSgeqrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverDgeqrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverCgeqrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverZgeqrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            tau,
                                         int               stT,
                                         float*            work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverSgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           tau,
                                         int               stT,
                                         double*           work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverDgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_geqrf(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         hipsolverComplex* tau,
                                         int               stT,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    if(!FORTRAN)
        return hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverCgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_geqrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* tau,
                                         int                     stT,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    else
        return hipsolverZgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesvd_bufferSize(bool              FORTRAN,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    float*            A,
                                                    int               lda,
                                                    int*              lwork)
{
    if(!FORTRAN)
        return hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    else
        return hipsolverSgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(bool              FORTRAN,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    double*           A,
                                                    int               lda,
                                                    int*              lwork)
{
    if(!FORTRAN)
        return hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    else
        return hipsolverDgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(bool              FORTRAN,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    hipsolverComplex* A,
                                                    int               lda,
                                                    int*              lwork)
{
    if(!FORTRAN)
        return hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    else
        return hipsolverCgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    signed char             jobu,
                                                    signed char             jobv,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    else
        return hipsolverZgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
}

inline hipsolverStatus_t hipsolver_gesvd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         float*            U,
                                         int               ldu,
                                         int               stU,
                                         float*            V,
                                         int               ldv,
                                         int               stV,
                                         float*            work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case FORTRAN_NORMAL:
        return hipsolverSgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           S,
                                         int               stS,
                                         double*           U,
                                         int               ldu,
                                         int               stU,
                                         double*           V,
                                         int               ldv,
                                         int               stV,
                                         double*           work,
                                         int               lwork,
                                         double*           rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case FORTRAN_NORMAL:
        return hipsolverDgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(bool              FORTRAN,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         hipsolverComplex* U,
                                         int               ldu,
                                         int               stU,
                                         hipsolverComplex* V,
                                         int               ldv,
                                         int               stV,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverCgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case FORTRAN_NORMAL:
        return hipsolverCgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         signed char             jobu,
                                         signed char             jobv,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 S,
                                         int                     stS,
                                         hipsolverDoubleComplex* U,
                                         int                     ldu,
                                         int                     stU,
                                         hipsolverDoubleComplex* V,
                                         int                     ldv,
                                         int                     stV,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         double*                 rwork,
                                         int                     stRW,
                                         int*                    info,
                                         int                     bc)
{
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverZgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case FORTRAN_NORMAL:
        return hipsolverZgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverSgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverDgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverCgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork);
    else
        return hipsolverZgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_getrf(bool              FORTRAN,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            work,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, NPVT))
    {
    case C_NORMAL:
        return hipsolverSgetrf(handle, m, n, A, lda, work, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverSgetrf(handle, m, n, A, lda, work, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverSgetrfFortran(handle, m, n, A, lda, work, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSgetrfFortran(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(bool              FORTRAN,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           work,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, NPVT))
    {
    case C_NORMAL:
        return hipsolverDgetrf(handle, m, n, A, lda, work, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverDgetrf(handle, m, n, A, lda, work, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverDgetrfFortran(handle, m, n, A, lda, work, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDgetrfFortran(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(bool              FORTRAN,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         hipsolverComplex* work,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(bool2marshal(FORTRAN, NPVT))
    {
    case C_NORMAL:
        return hipsolverCgetrf(handle, m, n, A, lda, work, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverCgetrf(handle, m, n, A, lda, work, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverCgetrfFortran(handle, m, n, A, lda, work, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCgetrfFortran(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(bool                    FORTRAN,
                                         bool                    NPVT,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int*                    ipiv,
                                         int                     stP,
                                         int*                    info,
                                         int                     bc)
{
    switch(bool2marshal(FORTRAN, NPVT))
    {
    case C_NORMAL:
        return hipsolverZgetrf(handle, m, n, A, lda, work, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverZgetrf(handle, m, n, A, lda, work, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverZgetrfFortran(handle, m, n, A, lda, work, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZgetrfFortran(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrs(bool                 FORTRAN,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         float*               A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         float*               B,
                                         int                  ldb,
                                         int                  stB,
                                         int*                 info,
                                         int                  bc)
{
    if(!FORTRAN)
        return hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    else
        return hipsolverSgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline hipsolverStatus_t hipsolver_getrs(bool                 FORTRAN,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         double*              A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         double*              B,
                                         int                  ldb,
                                         int                  stB,
                                         int*                 info,
                                         int                  bc)
{
    if(!FORTRAN)
        return hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    else
        return hipsolverDgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline hipsolverStatus_t hipsolver_getrs(bool                 FORTRAN,
                                         hipsolverHandle_t    handle,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         hipsolverComplex*    A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         hipsolverComplex*    B,
                                         int                  ldb,
                                         int                  stB,
                                         int*                 info,
                                         int                  bc)
{
    if(!FORTRAN)
        return hipsolverCgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    else
        return hipsolverCgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}

inline hipsolverStatus_t hipsolver_getrs(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverOperation_t    trans,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    else
        return hipsolverZgetrsFortran(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
/********************************************************/

/******************** POTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverSpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverDpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverCpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverZpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverCpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverZpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
}

// batched
inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverSpotrfBatched(handle, uplo, n, A, lda, info, bc);
    else
        return hipsolverSpotrfBatchedFortran(handle, uplo, n, A, lda, info, bc);
}

inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverDpotrfBatched(handle, uplo, n, A, lda, info, bc);
    else
        return hipsolverDpotrfBatchedFortran(handle, uplo, n, A, lda, info, bc);
}

inline hipsolverStatus_t hipsolver_potrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverCpotrfBatched(handle, uplo, n, A, lda, info, bc);
    else
        return hipsolverCpotrfBatchedFortran(handle, uplo, n, A, lda, info, bc);
}

inline hipsolverStatus_t hipsolver_potrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZpotrfBatched(handle, uplo, n, A, lda, info, bc);
    else
        return hipsolverZpotrfBatchedFortran(handle, uplo, n, A, lda, info, bc);
}
/********************************************************/

/******************** SYTRD/HETRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          float*              tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    else
        return hipsolverSsytrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             E,
                                                          double*             tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    else
        return hipsolverDsytrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    else
        return hipsolverChetrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          double*                 E,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    else
        return hipsolverZhetrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               float*              tau,
                                               int                 stP,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    else
        return hipsolverSsytrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             D,
                                               int                 stD,
                                               double*             E,
                                               int                 stE,
                                               double*             tau,
                                               int                 stP,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    else
        return hipsolverDsytrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               hipsolverComplex*   tau,
                                               int                 stP,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    else
        return hipsolverChetrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 D,
                                               int                     stD,
                                               double*                 E,
                                               int                     stE,
                                               hipsolverDoubleComplex* tau,
                                               int                     stP,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    if(!FORTRAN)
        return hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    else
        return hipsolverZhetrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
}
/********************************************************/
