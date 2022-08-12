/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-types.h defines data types used by hipsolver
 */

#ifndef HIPSOLVER_TYPES_H
#define HIPSOLVER_TYPES_H

#include <hip/hip_complex.h>

typedef void* hipsolverHandle_t;

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

typedef enum
{
    HIPSOLVER_SIDE_LEFT  = 141,
    HIPSOLVER_SIDE_RIGHT = 142,
} hipsolverSideMode_t;

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

#endif // HIPSOLVER_TYPES_H
