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

inline testMarshal_t bool2marshal(bool FORTRAN, bool STRIDED, bool ALT = false)
{
    if(!FORTRAN)
        if(!STRIDED)
            if(!ALT)
                return C_NORMAL;
            else
                return C_NORMAL_ALT;
        else if(!ALT)
            return C_STRIDED;
        else
            return C_STRIDED_ALT;
    else if(!STRIDED)
        if(!ALT)
            return FORTRAN_NORMAL;
        else
            return FORTRAN_NORMAL_ALT;
    else if(!ALT)
        return FORTRAN_STRIDED;
    else
        return FORTRAN_STRIDED_ALT;
}

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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverSgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverDgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverCgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverZgeqrfFortran(handle, m, n, A, lda, tau, work, lwork, info);
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
    switch(bool2marshal(FORTRAN, bc != 1, NPVT))
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
    switch(bool2marshal(FORTRAN, bc != 1, NPVT))
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
    switch(bool2marshal(FORTRAN, bc != 1, NPVT))
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
    switch(bool2marshal(FORTRAN, bc != 1, NPVT))
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverCpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
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
    switch(bool2marshal(FORTRAN, bc != 1))
    {
    case C_NORMAL:
        return hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case FORTRAN_NORMAL:
        return hipsolverZpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/
