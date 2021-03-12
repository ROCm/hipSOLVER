/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************/

#include "../include/lapack_host_reference.hpp"
#include "cblas.h"
#include "hipsolver.h"

/*!\file
 * \brief provide template functions interfaces to BLAS and LAPACK interfaces, it is
 * only used for testing, not part of the GPU library
 */

/*************************************************************************/
// These are C wrapper calls to CBLAS and fortran LAPACK

#ifdef __cplusplus
extern "C" {
#endif

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, hipsolverComplex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, hipsolverDoubleComplex* A, int* lda, int* ipiv, int* info);

#ifdef __cplusplus
}
#endif
/************************************************************************/

/************************************************************************/
// These are templated functions used in hipSOLVER clients code

// getrf
template <>
void cblas_getrf<float>(int m, int n, float* A, int lda, int* ipiv, int* info)
{
    sgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<double>(int m, int n, double* A, int lda, int* ipiv, int* info)
{
    dgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<hipsolverComplex>(int m, int n, hipsolverComplex* A, int lda, int* ipiv, int* info)
{
    cgetrf_(&m, &n, A, &lda, ipiv, info);
}

template <>
void cblas_getrf<hipsolverDoubleComplex>(
    int m, int n, hipsolverDoubleComplex* A, int lda, int* ipiv, int* info)
{
    zgetrf_(&m, &n, A, &lda, ipiv, info);
}
