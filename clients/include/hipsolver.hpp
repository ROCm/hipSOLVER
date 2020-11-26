/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef HIPSOLVER_HPP
#define HIPSOLVER_HPP

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

testMarshal_t bool2marshal(bool FORTRAN, bool STRIDED, bool ALT = false)
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

#endif
