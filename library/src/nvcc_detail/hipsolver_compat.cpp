/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief Implementation of the compatibility APIs that require especial calls
 *  to hipSOLVER on the cuSOLVER side.
 */

#include "hipsolver.h"

extern "C" {

// gesvd
hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverSgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverDgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverCgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverZgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

// getrf
hipsolverStatus_t hipsolverDnSgetrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              work,
                                    int*                devIpiv,
                                    int*                devInfo)
{
    return hipsolverSgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnDgetrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             work,
                                    int*                devIpiv,
                                    int*                devInfo)
{
    return hipsolverDgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnCgetrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    hipFloatComplex*    work,
                                    int*                devIpiv,
                                    int*                devInfo)
{
    return hipsolverCgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnZgetrf(hipsolverDnHandle_t handle,
                                    int                 m,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    hipDoubleComplex*   work,
                                    int*                devIpiv,
                                    int*                devInfo)
{
    return hipsolverZgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
}

} //extern C
