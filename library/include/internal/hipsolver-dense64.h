/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_DENSE64_H
#define HIPSOLVER_DENSE64_H

#include "hipsolver-dense.h"
#include "hipsolver-types.h"

typedef void* hipsolverDnParams_t;

typedef enum
{
    HIPSOLVER_ALG_0 = 231,
    HIPSOLVER_ALG_1 = 232,
} hipsolverAlgMode_t;

typedef enum
{
    HIPSOLVERDN_GETRF = 0,
} hipsolverDnFunction_t;

#ifdef __cplusplus
extern "C" {
#endif

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSetAdvOptions(hipsolverDnParams_t   params,
                                                            hipsolverDnFunction_t func,
                                                            hipsolverAlgMode_t    alg);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_DENSE64_H
