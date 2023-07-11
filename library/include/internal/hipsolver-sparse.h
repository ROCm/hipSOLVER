/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_SPARSE_H
#define HIPSOLVER_SPARSE_H

#include "hipsolver-types.h"

typedef void* hipsolverSpHandle_t;

typedef void* hipsparseMatDescr_t;

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle,
                                                        hipStream_t         streamId);

// linear solver based on Cholesky
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t       handle,
                                                          int                       n,
                                                          int                       nnzA,
                                                          const hipsparseMatDescr_t descrA,
                                                          const float*              csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          const float*              b,
                                                          float                     tolerance,
                                                          int                       reorder,
                                                          float*                    x,
                                                          int*                      singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t       handle,
                                                          int                       n,
                                                          int                       nnzA,
                                                          const hipsparseMatDescr_t descrA,
                                                          const double*             csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          const double*             b,
                                                          double                    tolerance,
                                                          int                       reorder,
                                                          double*                   x,
                                                          int*                      singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpCcsrlsvchol(hipsolverSpHandle_t       handle,
                                                          int                       n,
                                                          int                       nnzA,
                                                          const hipsparseMatDescr_t descrA,
                                                          const hipFloatComplex*    csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          const hipFloatComplex*    b,
                                                          float                     tolerance,
                                                          int                       reorder,
                                                          hipFloatComplex*          x,
                                                          int*                      singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpZcsrlsvchol(hipsolverSpHandle_t       handle,
                                                          int                       n,
                                                          int                       nnzA,
                                                          const hipsparseMatDescr_t descrA,
                                                          const hipDoubleComplex*   csrVal,
                                                          const int*                csrRowPtr,
                                                          const int*                csrColInd,
                                                          const hipDoubleComplex*   b,
                                                          double                    tolerance,
                                                          int                       reorder,
                                                          hipDoubleComplex*         x,
                                                          int*                      singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t       handle,
                                                              int                       n,
                                                              int                       nnzA,
                                                              const hipsparseMatDescr_t descrA,
                                                              const float*              csrVal,
                                                              const int*                csrRowPtr,
                                                              const int*                csrColInd,
                                                              const float*              b,
                                                              float                     tolerance,
                                                              int                       reorder,
                                                              float*                    x,
                                                              int* singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                                              int                       n,
                                                              int                       nnzA,
                                                              const hipsparseMatDescr_t descrA,
                                                              const double*             csrVal,
                                                              const int*                csrRowPtr,
                                                              const int*                csrColInd,
                                                              const double*             b,
                                                              double                    tolerance,
                                                              int                       reorder,
                                                              double*                   x,
                                                              int* singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpCcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                                              int                       n,
                                                              int                       nnzA,
                                                              const hipsparseMatDescr_t descrA,
                                                              const hipFloatComplex*    csrVal,
                                                              const int*                csrRowPtr,
                                                              const int*                csrColInd,
                                                              const hipFloatComplex*    b,
                                                              float                     tolerance,
                                                              int                       reorder,
                                                              hipFloatComplex*          x,
                                                              int* singularity);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpZcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                                              int                       n,
                                                              int                       nnzA,
                                                              const hipsparseMatDescr_t descrA,
                                                              const hipDoubleComplex*   csrVal,
                                                              const int*                csrRowPtr,
                                                              const int*                csrColInd,
                                                              const hipDoubleComplex*   b,
                                                              double                    tolerance,
                                                              int                       reorder,
                                                              hipDoubleComplex*         x,
                                                              int* singularity);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_SPARSE_H
