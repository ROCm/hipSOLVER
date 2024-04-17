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

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgetrs(hipsolverDnHandle_t  handle,
                                                     hipsolverDnParams_t  params,
                                                     hipsolverOperation_t trans,
                                                     int64_t              n,
                                                     int64_t              nrhs,
                                                     hipDataType          dataTypeA,
                                                     const void*          A,
                                                     int64_t              lda,
                                                     const int64_t*       devIpiv,
                                                     hipDataType          dataTypeB,
                                                     void*                B,
                                                     int64_t              ldb,
                                                     int*                 devInfo);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_DENSE64_H
