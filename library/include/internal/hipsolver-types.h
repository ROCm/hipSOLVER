/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-types.h defines data types used by hipsolver
 */

#ifndef HIPSOLVER_TYPES_H
#define HIPSOLVER_TYPES_H

#include <hip/hip_complex.h>

typedef void* hipsolverHandle_t;

typedef void* hipsolverGesvdjInfo_t;
typedef void* hipsolverSyevjInfo_t;

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
    HIPSOLVER_STATUS_ZERO_PIVOT        = 12,
} hipsolverStatus_t;

#ifndef HIPBLAS_SHARED_ENUMS
#define HIPBLAS_SHARED_ENUMS

typedef enum
{
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113,
} hipblasOperation_t;

typedef enum
{
    HIPBLAS_FILL_MODE_UPPER = 121,
    HIPBLAS_FILL_MODE_LOWER = 122,
    HIPBLAS_FILL_MODE_FULL  = 123,
} hipblasFillMode_t;

typedef enum
{
    HIPBLAS_DIAG_NON_UNIT = 131,
    HIPBLAS_DIAG_UNIT     = 132,
} hipblasDiagType_t;

typedef enum
{
    HIPBLAS_SIDE_LEFT  = 141,
    HIPBLAS_SIDE_RIGHT = 142,
    HIPBLAS_SIDE_BOTH  = 143,
} hipblasSideMode_t;

#endif // HIPBLAS_SHARED_ENUMS

typedef enum
{
    HIPSOLVER_EIG_MODE_NOVECTOR = 201,
    HIPSOLVER_EIG_MODE_VECTOR   = 202,
} hipsolverEigMode_t;

typedef enum
{
    HIPSOLVER_EIG_TYPE_1 = 211,
    HIPSOLVER_EIG_TYPE_2 = 212,
    HIPSOLVER_EIG_TYPE_3 = 213,
} hipsolverEigType_t;

// Aliases for hipBLAS enums
typedef hipblasOperation_t hipsolverOperation_t;
#define HIPSOLVER_OP_N HIPBLAS_OP_N
#define HIPSOLVER_OP_T HIPBLAS_OP_T
#define HIPSOLVER_OP_C HIPBLAS_OP_C

typedef hipblasFillMode_t hipsolverFillMode_t;
#define HIPSOLVER_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define HIPSOLVER_FILL_MODE_LOWER HIPBLAS_FILL_MODE_LOWER

typedef hipblasSideMode_t hipsolverSideMode_t;
#define HIPSOLVER_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define HIPSOLVER_SIDE_RIGHT HIPBLAS_SIDE_RIGHT

// Ensure hipBLAS enums have expected values
#if __cplusplus >= 201103L

static_assert(HIPBLAS_OP_N == 111);
static_assert(HIPBLAS_OP_T == 112);
static_assert(HIPBLAS_OP_C == 113);

static_assert(HIPBLAS_FILL_MODE_UPPER == 121);
static_assert(HIPBLAS_FILL_MODE_LOWER == 122);
static_assert(HIPBLAS_FILL_MODE_FULL == 123);

static_assert(HIPBLAS_DIAG_NON_UNIT == 131);
static_assert(HIPBLAS_DIAG_UNIT == 132);

static_assert(HIPBLAS_SIDE_LEFT == 141);
static_assert(HIPBLAS_SIDE_RIGHT == 142);
static_assert(HIPBLAS_SIDE_BOTH == 143);

#endif // __cplusplus

#endif // HIPSOLVER_TYPES_H
