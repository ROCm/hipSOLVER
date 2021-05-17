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

void sormqr_(char*  side,
             char*  trans,
             int*   m,
             int*   n,
             int*   k,
             float* A,
             int*   lda,
             float* ipiv,
             float* C,
             int*   ldc,
             float* work,
             int*   sizeW,
             int*   info);
void dormqr_(char*   side,
             char*   trans,
             int*    m,
             int*    n,
             int*    k,
             double* A,
             int*    lda,
             double* ipiv,
             double* C,
             int*    ldc,
             double* work,
             int*    sizeW,
             int*    info);
void cunmqr_(char*             side,
             char*             trans,
             int*              m,
             int*              n,
             int*              k,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* C,
             int*              ldc,
             hipsolverComplex* work,
             int*              sizeW,
             int*              info);
void zunmqr_(char*                   side,
             char*                   trans,
             int*                    m,
             int*                    n,
             int*                    k,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* C,
             int*                    ldc,
             hipsolverDoubleComplex* work,
             int*                    sizeW,
             int*                    info);

void sgeqrf_(int* m, int* n, float* A, int* lda, float* ipiv, float* work, int* lwork, int* info);
void dgeqrf_(
    int* m, int* n, double* A, int* lda, double* ipiv, double* work, int* lwork, int* info);
void cgeqrf_(int*              m,
             int*              n,
             hipsolverComplex* A,
             int*              lda,
             hipsolverComplex* ipiv,
             hipsolverComplex* work,
             int*              lwork,
             int*              info);
void zgeqrf_(int*                    m,
             int*                    n,
             hipsolverDoubleComplex* A,
             int*                    lda,
             hipsolverDoubleComplex* ipiv,
             hipsolverDoubleComplex* work,
             int*                    lwork,
             int*                    info);

void sgetrf_(int* m, int* n, float* A, int* lda, int* ipiv, int* info);
void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);
void cgetrf_(int* m, int* n, hipsolverComplex* A, int* lda, int* ipiv, int* info);
void zgetrf_(int* m, int* n, hipsolverDoubleComplex* A, int* lda, int* ipiv, int* info);

void sgetrs_(
    char* trans, int* n, int* nrhs, float* A, int* lda, int* ipiv, float* B, int* ldb, int* info);
void dgetrs_(
    char* trans, int* n, int* nrhs, double* A, int* lda, int* ipiv, double* B, int* ldb, int* info);
void cgetrs_(char*             trans,
             int*              n,
             int*              nrhs,
             hipsolverComplex* A,
             int*              lda,
             int*              ipiv,
             hipsolverComplex* B,
             int*              ldb,
             int*              info);
void zgetrs_(char*                   trans,
             int*                    n,
             int*                    nrhs,
             hipsolverDoubleComplex* A,
             int*                    lda,
             int*                    ipiv,
             hipsolverDoubleComplex* B,
             int*                    ldb,
             int*                    info);

void spotrf_(char* uplo, int* m, float* A, int* lda, int* info);
void dpotrf_(char* uplo, int* m, double* A, int* lda, int* info);
void cpotrf_(char* uplo, int* m, hipsolverComplex* A, int* lda, int* info);
void zpotrf_(char* uplo, int* m, hipsolverDoubleComplex* A, int* lda, int* info);

#ifdef __cplusplus
}
#endif
/************************************************************************/

/************************************************************************/
// These are templated functions used in hipSOLVER clients code

// gemm
template <>
void cblas_gemm<float>(hipsolverOperation_t transA,
                       hipsolverOperation_t transB,
                       int                  m,
                       int                  n,
                       int                  k,
                       float                alpha,
                       float*               A,
                       int                  lda,
                       float*               B,
                       int                  ldb,
                       float                beta,
                       float*               C,
                       int                  ldc)
{
    cblas_sgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<double>(hipsolverOperation_t transA,
                        hipsolverOperation_t transB,
                        int                  m,
                        int                  n,
                        int                  k,
                        double               alpha,
                        double*              A,
                        int                  lda,
                        double*              B,
                        int                  ldb,
                        double               beta,
                        double*              C,
                        int                  ldc)
{
    cblas_dgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                alpha,
                A,
                lda,
                B,
                ldb,
                beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipsolverComplex>(hipsolverOperation_t transA,
                                  hipsolverOperation_t transB,
                                  int                  m,
                                  int                  n,
                                  int                  k,
                                  hipsolverComplex     alpha,
                                  hipsolverComplex*    A,
                                  int                  lda,
                                  hipsolverComplex*    B,
                                  int                  ldb,
                                  hipsolverComplex     beta,
                                  hipsolverComplex*    C,
                                  int                  ldc)
{
    cblas_cgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

template <>
void cblas_gemm<hipsolverDoubleComplex>(hipsolverOperation_t    transA,
                                        hipsolverOperation_t    transB,
                                        int                     m,
                                        int                     n,
                                        int                     k,
                                        hipsolverDoubleComplex  alpha,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        hipsolverDoubleComplex  beta,
                                        hipsolverDoubleComplex* C,
                                        int                     ldc)
{
    cblas_zgemm(CblasColMajor,
                (CBLAS_TRANSPOSE)transA,
                (CBLAS_TRANSPOSE)transB,
                m,
                n,
                k,
                &alpha,
                A,
                lda,
                B,
                ldb,
                &beta,
                C,
                ldc);
}

// ormqr & unmqr
template <>
void cblas_ormqr_unmqr<float>(hipsolverSideMode_t  side,
                              hipsolverOperation_t trans,
                              int                  m,
                              int                  n,
                              int                  k,
                              float*               A,
                              int                  lda,
                              float*               ipiv,
                              float*               C,
                              int                  ldc,
                              float*               work,
                              int                  lwork)
{
    int  info;
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    sormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<double>(hipsolverSideMode_t  side,
                               hipsolverOperation_t trans,
                               int                  m,
                               int                  n,
                               int                  k,
                               double*              A,
                               int                  lda,
                               double*              ipiv,
                               double*              C,
                               int                  ldc,
                               double*              work,
                               int                  lwork)
{
    int  info;
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    dormqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<hipsolverComplex>(hipsolverSideMode_t  side,
                                         hipsolverOperation_t trans,
                                         int                  m,
                                         int                  n,
                                         int                  k,
                                         hipsolverComplex*    A,
                                         int                  lda,
                                         hipsolverComplex*    ipiv,
                                         hipsolverComplex*    C,
                                         int                  ldc,
                                         hipsolverComplex*    work,
                                         int                  lwork)
{
    int  info;
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    cunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

template <>
void cblas_ormqr_unmqr<hipsolverDoubleComplex>(hipsolverSideMode_t     side,
                                               hipsolverOperation_t    trans,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* ipiv,
                                               hipsolverDoubleComplex* C,
                                               int                     ldc,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork)
{
    int  info;
    char sideC  = hipsolver2char_side(side);
    char transC = hipsolver2char_operation(trans);

    zunmqr_(&sideC, &transC, &m, &n, &k, A, &lda, ipiv, C, &ldc, work, &lwork, &info);
}

// geqrf
template <>
void cblas_geqrf<float>(int m, int n, float* A, int lda, float* ipiv, float* work, int lwork)
{
    int info;
    sgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<double>(int m, int n, double* A, int lda, double* ipiv, double* work, int lwork)
{
    int info;
    dgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<hipsolverComplex>(int               m,
                                   int               n,
                                   hipsolverComplex* A,
                                   int               lda,
                                   hipsolverComplex* ipiv,
                                   hipsolverComplex* work,
                                   int               lwork)
{
    int info;
    cgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

template <>
void cblas_geqrf<hipsolverDoubleComplex>(int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         hipsolverDoubleComplex* ipiv,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork)
{
    int info;
    zgeqrf_(&m, &n, A, &lda, ipiv, work, &lwork, &info);
}

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

// getrs
template <>
void cblas_getrs<float>(
    hipsolverOperation_t trans, int n, int nrhs, float* A, int lda, int* ipiv, float* B, int ldb)
{
    int  info;
    char transC = hipsolver2char_operation(trans);
    sgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<double>(
    hipsolverOperation_t trans, int n, int nrhs, double* A, int lda, int* ipiv, double* B, int ldb)
{
    int  info;
    char transC = hipsolver2char_operation(trans);
    dgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<hipsolverComplex>(hipsolverOperation_t trans,
                                   int                  n,
                                   int                  nrhs,
                                   hipsolverComplex*    A,
                                   int                  lda,
                                   int*                 ipiv,
                                   hipsolverComplex*    B,
                                   int                  ldb)
{
    int  info;
    char transC = hipsolver2char_operation(trans);
    cgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

template <>
void cblas_getrs<hipsolverDoubleComplex>(hipsolverOperation_t    trans,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int*                    ipiv,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb)
{
    int  info;
    char transC = hipsolver2char_operation(trans);
    zgetrs_(&transC, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
}

// potrf
template <>
void cblas_potrf<float>(hipsolverFillMode_t uplo, int n, float* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    spotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf<double>(hipsolverFillMode_t uplo, int n, double* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    dpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf<hipsolverComplex>(
    hipsolverFillMode_t uplo, int n, hipsolverComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    cpotrf_(&uploC, &n, A, &lda, info);
}

template <>
void cblas_potrf<hipsolverDoubleComplex>(
    hipsolverFillMode_t uplo, int n, hipsolverDoubleComplex* A, int lda, int* info)
{
    char uploC = hipsolver2char_fill(uplo);
    zpotrf_(&uploC, &n, A, &lda, info);
}
