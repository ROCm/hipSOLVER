/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#ifdef _WIN32
#include "hipsolver_no_fortran.hpp"
#else
#include "hipsolver_fortran.hpp"
#endif

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
    API_NORMAL,
    API_FORTRAN,
    API_COMPAT
} testAPI_t;

typedef enum
{
    C_NORMAL,
    C_NORMAL_ALT,
    C_STRIDED,
    C_STRIDED_ALT,
    FORTRAN_NORMAL,
    FORTRAN_NORMAL_ALT,
    FORTRAN_STRIDED,
    FORTRAN_STRIDED_ALT,
    COMPAT_NORMAL,
    COMPAT_NORMAL_ALT,
    COMPAT_STRIDED,
    COMPAT_STRIDED_ALT,
    INVALID_API_SPEC
} testMarshal_t;

inline testMarshal_t api2marshal(testAPI_t API, bool ALT)
{
    switch(API)
    {
    case API_NORMAL:
        if(!ALT)
            return C_NORMAL;
        else
            return C_NORMAL_ALT;
    case API_FORTRAN:
        if(!ALT)
            return FORTRAN_NORMAL;
        else
            return FORTRAN_NORMAL_ALT;
    case API_COMPAT:
        if(!ALT)
            return COMPAT_NORMAL;
        else
            return COMPAT_NORMAL_ALT;
    default:
        return INVALID_API_SPEC;
    }
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
        return hipsolverCungbr_bufferSize(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    else
        return hipsolverCungbr_bufferSizeFortran(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
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
        return hipsolverZungbr_bufferSize(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    else
        return hipsolverZungbr_bufferSizeFortran(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
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
        return hipsolverCungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCungbrFortran(handle,
                                      side,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZungbrFortran(handle,
                                      side,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverCungqr_bufferSize(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    else
        return hipsolverCungqr_bufferSizeFortran(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
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
        return hipsolverZungqr_bufferSize(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    else
        return hipsolverZungqr_bufferSizeFortran(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
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
        return hipsolverCungqr(handle,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCungqrFortran(handle,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZungqr(handle,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZungqrFortran(handle,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverCungtr_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    else
        return hipsolverCungtr_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
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
        return hipsolverZungtr_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    else
        return hipsolverZungtr_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
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
        return hipsolverCungtr(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCungtrFortran(handle,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZungtr(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZungtrFortran(handle,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverCunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);
    else
        return hipsolverCunmqr_bufferSizeFortran(handle,
                                                 side,
                                                 trans,
                                                 m,
                                                 n,
                                                 k,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)tau,
                                                 (hipFloatComplex*)C,
                                                 ldc,
                                                 lwork);
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
        return hipsolverZunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);
    else
        return hipsolverZunmqr_bufferSizeFortran(handle,
                                                 side,
                                                 trans,
                                                 m,
                                                 n,
                                                 k,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)tau,
                                                 (hipDoubleComplex*)C,
                                                 ldc,
                                                 lwork);
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
        return hipsolverCunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCunmqrFortran(handle,
                                      side,
                                      trans,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)C,
                                      ldc,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZunmqrFortran(handle,
                                      side,
                                      trans,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
}
/********************************************************/

/******************** ORMTR/UNMTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
    if(!FORTRAN)
        return hipsolverSormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverSormtr_bufferSizeFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
    if(!FORTRAN)
        return hipsolverDormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    else
        return hipsolverDormtr_bufferSizeFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                 FORTRAN,
                                                          hipsolverHandle_t    handle,
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
                                                          int*                 lwork)
{
    if(!FORTRAN)
        return hipsolverCunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);
    else
        return hipsolverCunmtr_bufferSizeFortran(handle,
                                                 side,
                                                 uplo,
                                                 trans,
                                                 m,
                                                 n,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)tau,
                                                 (hipFloatComplex*)C,
                                                 ldc,
                                                 lwork);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
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
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);
    else
        return hipsolverZunmtr_bufferSizeFortran(handle,
                                                 side,
                                                 uplo,
                                                 trans,
                                                 m,
                                                 n,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)tau,
                                                 (hipDoubleComplex*)C,
                                                 ldc,
                                                 lwork);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverSormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverSormtrFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverDormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    else
        return hipsolverDormtrFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                 FORTRAN,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    if(!FORTRAN)
        return hipsolverCunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCunmtrFortran(handle,
                                      side,
                                      uplo,
                                      trans,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)C,
                                      ldc,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
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
                                               int*                    info)
{
    if(!FORTRAN)
        return hipsolverZunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZunmtrFortran(handle,
                                      side,
                                      uplo,
                                      trans,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverCgebrd(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tauq,
                               (hipFloatComplex*)taup,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCgebrdFortran(handle,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipFloatComplex*)tauq,
                                      (hipFloatComplex*)taup,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZgebrd(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tauq,
                               (hipDoubleComplex*)taup,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZgebrdFortran(handle,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipDoubleComplex*)tauq,
                                      (hipDoubleComplex*)taup,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverSSgels_bufferSizeFortran(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnSSgels_bufferSize(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverDDgels_bufferSizeFortran(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnDDgels_bufferSize(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCCgels_bufferSizeFortran(handle,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 (hipFloatComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCCgels_bufferSize(handle,
                                            m,
                                            n,
                                            nrhs,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            (hipFloatComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     m,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZZgels_bufferSizeFortran(handle,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 (hipDoubleComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZZgels_bufferSize(handle,
                                            m,
                                            n,
                                            nrhs,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            (hipDoubleComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverSSgelsFortran(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverDDgelsFortran(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCCgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCCgelsFortran(handle,
                                      m,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCCgels(handle,
                                 m,
                                 n,
                                 nrhs,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 (hipFloatComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
                                        int                     m,
                                        int                     n,
                                        int                     nrhs,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        int                     stA,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        int                     stB,
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZZgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZZgelsFortran(handle,
                                      m,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZZgels(handle,
                                 m,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 (hipDoubleComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
        return hipsolverCgeqrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lwork);
    else
        return hipsolverCgeqrf_bufferSizeFortran(handle, m, n, (hipFloatComplex*)A, lda, lwork);
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
        return hipsolverZgeqrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lwork);
    else
        return hipsolverZgeqrf_bufferSizeFortran(handle, m, n, (hipDoubleComplex*)A, lda, lwork);
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
        return hipsolverCgeqrf(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCgeqrfFortran(handle,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZgeqrf(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZgeqrfFortran(handle,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
}
/********************************************************/

/******************** GESV ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverSSgesv_bufferSizeFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnSSgesv_bufferSize(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverDDgesv_bufferSizeFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnDDgesv_bufferSize(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCCgesv_bufferSizeFortran(handle,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 (hipFloatComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCCgesv_bufferSize(handle,
                                            n,
                                            nrhs,
                                            (hipFloatComplex*)A,
                                            lda,
                                            ipiv,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            (hipFloatComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   int*                    ipiv,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZZgesv_bufferSizeFortran(handle,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 (hipDoubleComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZZgesv_bufferSize(handle,
                                            n,
                                            nrhs,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            ipiv,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            (hipDoubleComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverSSgesvFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverDDgesvFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgesv(handle,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCCgesv(handle,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCCgesvFortran(handle,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCCgesv(handle,
                                 n,
                                 nrhs,
                                 (hipFloatComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 (hipFloatComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
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
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgesv(handle,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZZgesv(handle,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZZgesvFortran(handle,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZZgesv(handle,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 (hipDoubleComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    float*            A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverSgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnSgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    double*           A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverDgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnDgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    hipsolverComplex* A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverCgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnCgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    signed char             jobu,
                                                    signed char             jobv,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverZgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnZgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case API_FORTRAN:
        return hipsolverSgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case API_COMPAT:
        return hipsolverDnSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case API_FORTRAN:
        return hipsolverDgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case API_COMPAT:
        return hipsolverDnDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               S,
                               (hipFloatComplex*)U,
                               ldu,
                               (hipFloatComplex*)V,
                               ldv,
                               (hipFloatComplex*)work,
                               lwork,
                               rwork,
                               info);
    case API_FORTRAN:
        return hipsolverCgesvdFortran(handle,
                                      jobu,
                                      jobv,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      S,
                                      (hipFloatComplex*)U,
                                      ldu,
                                      (hipFloatComplex*)V,
                                      ldv,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      rwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCgesvd(handle,
                                 jobu,
                                 jobv,
                                 m,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 S,
                                 (hipFloatComplex*)U,
                                 ldu,
                                 (hipFloatComplex*)V,
                                 ldv,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 rwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t               API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               S,
                               (hipDoubleComplex*)U,
                               ldu,
                               (hipDoubleComplex*)V,
                               ldv,
                               (hipDoubleComplex*)work,
                               lwork,
                               rwork,
                               info);
    case API_FORTRAN:
        return hipsolverZgesvdFortran(handle,
                                      jobu,
                                      jobv,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      S,
                                      (hipDoubleComplex*)U,
                                      ldu,
                                      (hipDoubleComplex*)V,
                                      ldv,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      rwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZgesvd(handle,
                                 jobu,
                                 jobv,
                                 m,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 S,
                                 (hipDoubleComplex*)U,
                                 ldu,
                                 (hipDoubleComplex*)V,
                                 ldv,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 rwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverSgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverDgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverCgetrf_bufferSizeFortran(handle, m, n, (hipFloatComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverZgetrf_bufferSizeFortran(handle, m, n, (hipDoubleComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverSgetrf(handle, m, n, A, lda, work, lwork, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverSgetrf(handle, m, n, A, lda, work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverSgetrfFortran(handle, m, n, A, lda, work, lwork, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSgetrfFortran(handle, m, n, A, lda, work, lwork, nullptr, info);
    case COMPAT_NORMAL:
        return hipsolverDnSgetrf(handle, m, n, A, lda, work, ipiv, info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSgetrf(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverDgetrf(handle, m, n, A, lda, work, lwork, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverDgetrf(handle, m, n, A, lda, work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverDgetrfFortran(handle, m, n, A, lda, work, lwork, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDgetrfFortran(handle, m, n, A, lda, work, lwork, nullptr, info);
    case COMPAT_NORMAL:
        return hipsolverDnDgetrf(handle, m, n, A, lda, work, ipiv, info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDgetrf(handle, m, n, A, lda, work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t         API,
                                         bool              NPVT,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              ipiv,
                                         int               stP,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverCgetrfFortran(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCgetrfFortran(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, nullptr, info);
    case COMPAT_NORMAL:
        return hipsolverDnCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, ipiv, info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t               API,
                                         bool                    NPVT,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    ipiv,
                                         int                     stP,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverZgetrfFortran(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, ipiv, info);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZgetrfFortran(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, nullptr, info);
    case COMPAT_NORMAL:
        return hipsolverDnZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, ipiv, info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, nullptr, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    float*               A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    float*               B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverSgetrs_bufferSizeFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    double*              A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    double*              B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverDgetrs_bufferSizeFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    hipsolverComplex*    A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    hipsolverComplex*    B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverOperation_t    trans,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    ipiv,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
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
                                         float*               work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
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
                                         double*              work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
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
                                         hipsolverComplex*    work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCgetrs(
            handle, trans, n, nrhs, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t               API,
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
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZgetrs(handle,
                                 trans,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverSpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverDpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverCpotrf_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverZpotrf_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverCpotrfFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnCpotrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverZpotrfFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnZpotrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverSpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverDpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched_bufferSize(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverCpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched_bufferSize(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverZpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverSpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnSpotrfBatched(handle, uplo, n, A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverDpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnDpotrfBatched(handle, uplo, n, A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverCpotrfBatchedFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnCpotrfBatched(handle, uplo, n, (hipFloatComplex**)A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
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
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverZpotrfBatchedFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnZpotrfBatched(handle, uplo, n, (hipDoubleComplex**)A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRI ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverSpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    else
        return hipsolverDpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                FORTRAN,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork)
{
    if(!FORTRAN)
        return hipsolverCpotri_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    else
        return hipsolverCpotri_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(bool                    FORTRAN,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZpotri_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    else
        return hipsolverZpotri_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
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
        return hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverSpotriFortran(handle, uplo, n, A, lda, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
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
        return hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, info);
    else
        return hipsolverDpotriFortran(handle, uplo, n, A, lda, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potri(bool                FORTRAN,
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
        return hipsolverCpotri(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    else
        return hipsolverCpotriFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
}

inline hipsolverStatus_t hipsolver_potri(bool                    FORTRAN,
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
        return hipsolverZpotri(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    else
        return hipsolverZpotriFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
}
/********************************************************/

/******************** POTRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A,
                                                    int                 lda,
                                                    float*              B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverSpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A,
                                                    int                 lda,
                                                    double*             B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverDpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    hipsolverComplex*   B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverCpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverZpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              B,
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             B,
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B,
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCpotrs(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZpotrs(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A[],
                                                    int                 lda,
                                                    float*              B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverSpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A[],
                                                    int                 lda,
                                                    double*             B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverDpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    hipsolverComplex*   B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverCpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B[],
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched_bufferSize(handle,
                                                 uplo,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex**)A,
                                                 lda,
                                                 (hipDoubleComplex**)B,
                                                 ldb,
                                                 lwork,
                                                 bc);
    case API_FORTRAN:
        return hipsolverZpotrsBatched_bufferSizeFortran(handle,
                                                        uplo,
                                                        n,
                                                        nrhs,
                                                        (hipDoubleComplex**)A,
                                                        lda,
                                                        (hipDoubleComplex**)B,
                                                        ldb,
                                                        lwork,
                                                        bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              B[],
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverSpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             B[],
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverDpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B[],
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex**)A,
                                      lda,
                                      (hipFloatComplex**)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      bc);
    case API_FORTRAN:
        return hipsolverCpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipFloatComplex**)A,
                                             lda,
                                             (hipFloatComplex**)B,
                                             ldb,
                                             (hipFloatComplex*)work,
                                             lwork,
                                             info,
                                             bc);
    case API_COMPAT:
        return hipsolverDnCpotrsBatched(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B[],
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex**)A,
                                      lda,
                                      (hipDoubleComplex**)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      bc);
    case API_FORTRAN:
        return hipsolverZpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipDoubleComplex**)A,
                                             lda,
                                             (hipDoubleComplex**)B,
                                             ldb,
                                             (hipDoubleComplex*)work,
                                             lwork,
                                             info,
                                             bc);
    case API_COMPAT:
        return hipsolverDnZpotrsBatched(handle,
                                        uplo,
                                        n,
                                        nrhs,
                                        (hipDoubleComplex**)A,
                                        lda,
                                        (hipDoubleComplex**)B,
                                        ldb,
                                        info,
                                        bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVD/HEEVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
    else
        return hipsolverSsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, D, lwork);
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork);
    else
        return hipsolverDsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, D, lwork);
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverCheevd_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, D, lwork);
    else
        return hipsolverCheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, D, lwork);
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZheevd_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, D, lwork);
    else
        return hipsolverZheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, D, lwork);
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info);
    else
        return hipsolverSsyevdFortran(handle, jobz, uplo, n, A, lda, D, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             D,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info);
    else
        return hipsolverDsyevdFortran(handle, jobz, uplo, n, A, lda, D, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverCheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverCheevdFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      D,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
}

inline hipsolverStatus_t hipsolver_syevd_heevd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 D,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    if(!FORTRAN)
        return hipsolverZheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZheevdFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      D,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
}
/********************************************************/

/******************** SYGVD/HEGVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
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
    if(!FORTRAN)
        return hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
    else
        return hipsolverSsygvd_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
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
    if(!FORTRAN)
        return hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
    else
        return hipsolverDsygvd_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                FORTRAN,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   B,
                                                          int                 ldb,
                                                          float*              D,
                                                          int*                lwork)
{
    if(!FORTRAN)
        return hipsolverChegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          D,
                                          lwork);
    else
        return hipsolverChegvd_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 D,
                                                 lwork);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(bool                    FORTRAN,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 D,
                                                          int*                    lwork)
{
    if(!FORTRAN)
        return hipsolverZhegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          D,
                                          lwork);
    else
        return hipsolverZhegvd_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 D,
                                                 lwork);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              D,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info);
    else
        return hipsolverSsygvdFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             B,
                                               int                 ldb,
                                               int                 stB,
                                               double*             D,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info);
    else
        return hipsolverDsygvdFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                FORTRAN,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               hipsolverComplex*   B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              D,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    if(!FORTRAN)
        return hipsolverChegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               D,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverChegvdFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      D,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(bool                    FORTRAN,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigType_t      itype,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               hipsolverDoubleComplex* B,
                                               int                     ldb,
                                               int                     stB,
                                               double*                 D,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    if(!FORTRAN)
        return hipsolverZhegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               D,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZhegvdFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      D,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverChetrd_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);
    else
        return hipsolverChetrd_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);
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
        return hipsolverZhetrd_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);
    else
        return hipsolverZhetrd_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);
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
        return hipsolverChetrd(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverChetrdFortran(handle,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
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
        return hipsolverZhetrd(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    else
        return hipsolverZhetrdFortran(handle,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
}
/********************************************************/

/******************** SYTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, float* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork);
    else
        return hipsolverSsytrf_bufferSizeFortran(handle, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, double* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork);
    else
        return hipsolverDsytrf_bufferSizeFortran(handle, n, A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, hipsolverComplex* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverCsytrf_bufferSize(handle, n, (hipFloatComplex*)A, lda, lwork);
    else
        return hipsolverCsytrf_bufferSizeFortran(handle, n, (hipFloatComplex*)A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    bool FORTRAN, hipsolverHandle_t handle, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{
    if(!FORTRAN)
        return hipsolverZsytrf_bufferSize(handle, n, (hipDoubleComplex*)A, lda, lwork);
    else
        return hipsolverZsytrf_bufferSizeFortran(handle, n, (hipDoubleComplex*)A, lda, lwork);
}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    else
        return hipsolverSsytrfFortran(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    else
        return hipsolverDsytrfFortran(handle, uplo, n, A, lda, ipiv, work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrf(bool                FORTRAN,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    if(!FORTRAN)
        return hipsolverCsytrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);
    else
        return hipsolverCsytrfFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);
}

inline hipsolverStatus_t hipsolver_sytrf(bool                    FORTRAN,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    if(!FORTRAN)
        return hipsolverZsytrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);
    else
        return hipsolverZsytrfFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);
}
/********************************************************/
