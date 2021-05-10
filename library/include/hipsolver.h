/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//! This is the master include file for hipsolver, wrapping around rocsolver and cusolver
//
#ifndef HIPSOLVER_H
#define HIPSOLVER_H

#include "hipsolver-export.h"
#include "hipsolver-version.h"
#include <hip/hip_runtime_api.h>
#include <stdint.h>

/* Workaround clang bug:

   https://bugs.llvm.org/show_bug.cgi?id=35863

   This macro expands to static if clang is used; otherwise it expands empty.
   It is intended to be used in variable template specializations, where clang
   requires static in order for the specializations to have internal linkage,
   while technically, storage class specifiers besides thread_local are not
   allowed in template specializations, and static in the primary template
   definition should imply internal linkage for all specializations.

   If clang shows an error for improperly using a storage class specifier in
   a specialization, then HIPSOLVER_CLANG_STATIC should be redefined as empty,
   and perhaps removed entirely, if the above bug has been fixed.
*/
#if __clang__
#define HIPSOLVER_CLANG_STATIC static
#else
#define HIPSOLVER_CLANG_STATIC
#endif

typedef void* hipsolverHandle_t;

typedef struct hipsolverComplex
{
#ifndef __cplusplus

    float x, y;

#else

private:
    float x, y;

public:
#if __cplusplus >= 201103L
    hipsolverComplex() = default;
#else
    hipsolverComplex() {}
#endif

    hipsolverComplex(float r, float i = 0)
        : x(r)
        , y(i)
    {
    }

    float real() const
    {
        return x;
    }
    float imag() const
    {
        return y;
    }
    void real(float r)
    {
        x = r;
    }
    void imag(float i)
    {
        y = i;
    }

#endif
} hipsolverComplex;

typedef struct hipsolverDoubleComplex
{
#ifndef __cplusplus

    double x, y;

#else

private:
    double x, y;

public:

#if __cplusplus >= 201103L
    hipsolverDoubleComplex() = default;
#else
    hipsolverDoubleComplex() {}
#endif

    hipsolverDoubleComplex(double r, double i = 0)
        : x(r)
        , y(i)
    {
    }

    double real() const
    {
        return x;
    }
    double imag() const
    {
        return y;
    }
    void real(double r)
    {
        x = r;
    }
    void imag(double i)
    {
        y = i;
    }

#endif
} hipsolverDoubleComplex;

#if __cplusplus >= 201103L
#include <type_traits>
static_assert(std::is_standard_layout<hipsolverComplex>{},
              "hipsolverComplex is not a standard layout type, and thus is incompatible with C.");
static_assert(
    std::is_standard_layout<hipsolverDoubleComplex>{},
    "hipsolverDoubleComplex is not a standard layout type, and thus is incompatible with C.");
static_assert(std::is_trivial<hipsolverComplex>{},
              "hipsolverComplex is not a trivial type, and thus is incompatible with C.");
static_assert(std::is_trivial<hipsolverDoubleComplex>{},
              "hipsolverDoubleComplex is not a trivial type, and thus is incompatible with C.");
static_assert(sizeof(hipsolverComplex) == sizeof(float) * 2
                  && sizeof(hipsolverDoubleComplex) == sizeof(double) * 2
                  && sizeof(hipsolverDoubleComplex) == sizeof(hipsolverComplex) * 2,
              "Sizes of hipsolverComplex or hipsolverDoubleComplex are inconsistent");
#endif

typedef enum
{
    HIPSOLVER_STATUS_SUCCESS           = 0, // Function succeeds
    HIPSOLVER_STATUS_NOT_INITIALIZED   = 1, // hipSOLVER library not initialized
    HIPSOLVER_STATUS_ALLOC_FAILED      = 2, // resource allocation failed
    HIPSOLVER_STATUS_INVALID_VALUE     = 3, // unsupported numerical value was passed to function
    HIPSOLVER_STATUS_MAPPING_ERROR     = 4, // access to GPU memory space failed
    HIPSOLVER_STATUS_EXECUTION_FAILED  = 5, // GPU program failed to execute
    HIPSOLVER_STATUS_INTERNAL_ERROR    = 6, // an internal hipSOLVER operation failed
    HIPSOLVER_STATUS_NOT_SUPPORTED     = 7, // function not implemented
    HIPSOLVER_STATUS_ARCH_MISMATCH     = 8,
    HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9, // hipSOLVER handle is null pointer
    HIPSOLVER_STATUS_INVALID_ENUM      = 10, // unsupported enum value was passed to function
    HIPSOLVER_STATUS_UNKNOWN           = 11, // back-end returned an unsupported status code
} hipsolverStatus_t;

// set the values of enum constants to be the same as those used in cblas
typedef enum
{
    HIPSOLVER_OP_N = 111,
    HIPSOLVER_OP_T = 112,
    HIPSOLVER_OP_C = 113,
} hipsolverOperation_t;

typedef enum
{
    HIPSOLVER_FILL_MODE_UPPER = 121,
    HIPSOLVER_FILL_MODE_LOWER = 122,
} hipsolverFillMode_t;

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCreate(hipsolverHandle_t* handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDestroy(hipsolverHandle_t handle);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSetStream(hipsolverHandle_t handle,
                                                      hipStream_t       streamId);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverGetStream(hipsolverHandle_t handle,
                                                      hipStream_t*      streamId);

// geqrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   float*            A,
                                                   int               lda,
                                                   float*            tau,
                                                   float*            work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           tau,
                                                   double*           work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   hipsolverComplex* tau,
                                                   hipsolverComplex* work,
                                                   int               lwork,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf(hipsolverHandle_t       handle,
                                                   int                     m,
                                                   int                     n,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* tau,
                                                   hipsolverDoubleComplex* work,
                                                   int                     lwork,
                                                   int*                    devInfo);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipsolverDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   float*            A,
                                                   int               lda,
                                                   float*            work,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   double*           A,
                                                   int               lda,
                                                   double*           work,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf(hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   hipsolverComplex* work,
                                                   int*              devIpiv,
                                                   int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf(hipsolverHandle_t       handle,
                                                   int                     m,
                                                   int                     n,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* work,
                                                   int*                    devIpiv,
                                                   int*                    devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   float*               A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   float*               B,
                                                   int                  ldb,
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   double*              A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   double*              B,
                                                   int                  ldb,
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrs(hipsolverHandle_t    handle,
                                                   hipsolverOperation_t trans,
                                                   int                  n,
                                                   int                  nrhs,
                                                   hipsolverComplex*    A,
                                                   int                  lda,
                                                   int*                 devIpiv,
                                                   hipsolverComplex*    B,
                                                   int                  ldb,
                                                   int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrs(hipsolverHandle_t       handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf_bufferSize(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf_bufferSize(hipsolverHandle_t   handle,
                                                              hipsolverFillMode_t uplo,
                                                              int                 n,
                                                              hipsolverComplex*   A,
                                                              int                 lda,
                                                              int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf_bufferSize(hipsolverHandle_t       handle,
                                                              hipsolverFillMode_t     uplo,
                                                              int                     n,
                                                              hipsolverDoubleComplex* A,
                                                              int                     lda,
                                                              int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   float*              A,
                                                   int                 lda,
                                                   float*              work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   double*             A,
                                                   int                 lda,
                                                   double*             work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf(hipsolverHandle_t   handle,
                                                   hipsolverFillMode_t uplo,
                                                   int                 n,
                                                   hipsolverComplex*   A,
                                                   int                 lda,
                                                   hipsolverComplex*   work,
                                                   int                 lwork,
                                                   int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf(hipsolverHandle_t       handle,
                                                   hipsolverFillMode_t     uplo,
                                                   int                     n,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* work,
                                                   int                     lwork,
                                                   int*                    devInfo);

#ifdef __cplusplus
}
#endif

#endif
