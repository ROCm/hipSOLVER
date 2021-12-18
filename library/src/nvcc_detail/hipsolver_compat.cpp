/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

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

hipsolverStatus_t hipsolverDnSgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    float*              A,
                                    int                 lda,
                                    float*              S,
                                    float*              U,
                                    int                 ldu,
                                    float*              V,
                                    int                 ldv,
                                    float*              work,
                                    int                 lwork,
                                    float*              rwork,
                                    int*                devInfo)
{
    return hipsolverSgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnDgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    double*             A,
                                    int                 lda,
                                    double*             S,
                                    double*             U,
                                    int                 ldu,
                                    double*             V,
                                    int                 ldv,
                                    double*             work,
                                    int                 lwork,
                                    double*             rwork,
                                    int*                devInfo)
{
    return hipsolverDgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnCgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    hipFloatComplex*    A,
                                    int                 lda,
                                    float*              S,
                                    hipFloatComplex*    U,
                                    int                 ldu,
                                    hipFloatComplex*    V,
                                    int                 ldv,
                                    hipFloatComplex*    work,
                                    int                 lwork,
                                    float*              rwork,
                                    int*                devInfo)
{
    return hipsolverCgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

hipsolverStatus_t hipsolverDnZgesvd(hipsolverDnHandle_t handle,
                                    signed char         jobu,
                                    signed char         jobv,
                                    int                 m,
                                    int                 n,
                                    hipDoubleComplex*   A,
                                    int                 lda,
                                    double*             S,
                                    hipDoubleComplex*   U,
                                    int                 ldu,
                                    hipDoubleComplex*   V,
                                    int                 ldv,
                                    hipDoubleComplex*   work,
                                    int                 lwork,
                                    double*             rwork,
                                    int*                devInfo)
{
    return hipsolverZgesvd(
        handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, devInfo);
}

// getrf
hipsolverStatus_t hipsolverDnSgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnDgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnCgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork)
{
    return hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

hipsolverStatus_t hipsolverDnZgetrf_bufferSize(
    hipsolverDnHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork)
{
    return hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork);
}

} //extern C
