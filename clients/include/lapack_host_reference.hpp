/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once

#include "hipsolver.h"
#include "hipsolver_datatype2string.hpp"

template <typename T>
void cblas_gemm(hipsolverOperation_t transA,
                hipsolverOperation_t transB,
                int                  m,
                int                  n,
                int                  k,
                T                    alpha,
                T*                   A,
                int                  lda,
                T*                   B,
                int                  ldb,
                T                    beta,
                T*                   C,
                int                  ldc);

template <typename T>
void cblas_orgqr_ungqr(int m, int n, int k, T* A, int lda, T* Ipiv, T* work, int sizeW);

template <typename T>
void cblas_ormqr_unmqr(hipsolverSideMode_t  side,
                       hipsolverOperation_t trans,
                       int                  m,
                       int                  n,
                       int                  k,
                       T*                   A,
                       int                  lda,
                       T*                   Ipiv,
                       T*                   C,
                       int                  ldc,
                       T*                   work,
                       int                  sizeW);

template <typename T>
void cblas_geqrf(int m, int n, T* A, int lda, T* ipiv, T* work, int sizeW);

template <typename T>
void cblas_getrf(int m, int n, T* A, int lda, int* ipiv, int* info);

template <typename T>
void cblas_getrs(
    hipsolverOperation_t trans, int n, int nrhs, T* A, int lda, int* ipiv, T* B, int ldb);

template <typename T>
void cblas_potrf(hipsolverFillMode_t uplo, int n, T* A, int lda, int* info);
