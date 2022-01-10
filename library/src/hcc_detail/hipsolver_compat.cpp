/* ************************************************************************
 * Copyright 2020-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief Implementation of the compatibility APIs that require especial calls
 *  to hipSOLVER on the rocSOLVER side.
 */

#include "error_macros.hpp"
#include "hipsolver.h"
#include <algorithm>
#include <iostream>

extern "C" {

// gesvd
hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverDnHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
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
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverSgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverSgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
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
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverDgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverDgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
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
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverCgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverCgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
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
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverZgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverZgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
}

} //extern C
