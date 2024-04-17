/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

/*! \file
 *  \brief Implementation of the compatibility APIs that require especial calls
 *  to hipSOLVER on the rocSOLVER side.
 */

#include "error_macros.hpp"
#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"
#include "utility.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

using namespace std;

extern "C" {

/******************** PARAMS ********************/
struct hipsolverParams
{
    hipsolverDnFunction_t func;
    hipsolverAlgMode_t    alg;

    // Constructor
    explicit hipsolverParams()
        : func(HIPSOLVERDN_GETRF)
        , alg(HIPSOLVER_ALG_0)
    {
    }
};

hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *info = new hipsolverParams;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverParams* params = (hipsolverParams*)info;
    delete params;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnSetAdvOptions(hipsolverDnParams_t   params,
                                           hipsolverDnFunction_t func,
                                           hipsolverAlgMode_t    alg)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverInternalXgetrs_bufferSize(hipsolverHandle_t    handle,
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
                                                     size_t*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;
    if(dataTypeA == HIP_R_32F && dataTypeB == HIP_R_32F)
    {
        status = rocblas2hip_status(rocsolver_sgetrs_64((rocblas_handle)handle,
                                                        hip2rocblas_operation(trans),
                                                        n,
                                                        nrhs,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldb));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeB == HIP_R_64F)
    {
        status = rocblas2hip_status(rocsolver_dgetrs_64((rocblas_handle)handle,
                                                        hip2rocblas_operation(trans),
                                                        n,
                                                        nrhs,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldb));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeB == HIP_C_32F)
    {
        status = rocblas2hip_status(rocsolver_cgetrs_64((rocblas_handle)handle,
                                                        hip2rocblas_operation(trans),
                                                        n,
                                                        nrhs,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldb));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeB == HIP_C_64F)
    {
        status = rocblas2hip_status(rocsolver_zgetrs_64((rocblas_handle)handle,
                                                        hip2rocblas_operation(trans),
                                                        n,
                                                        nrhs,
                                                        nullptr,
                                                        lda,
                                                        nullptr,
                                                        nullptr,
                                                        ldb));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lwork);

    return status;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnXgetrs(hipsolverDnHandle_t  handle,
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
                                    int*                 devInfo)
try
{
    size_t lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverInternalXgetrs_bufferSize((rocblas_handle)handle,
                                                             params,
                                                             trans,
                                                             n,
                                                             nrhs,
                                                             dataTypeA,
                                                             A,
                                                             lda,
                                                             devIpiv,
                                                             dataTypeB,
                                                             B,
                                                             ldb,
                                                             &lwork));
    CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    if(dataTypeA == HIP_R_32F && dataTypeB == HIP_R_32F)
    {
        return rocblas2hip_status(rocsolver_sgetrs_64((rocblas_handle)handle,
                                                      hip2rocblas_operation(trans),
                                                      n,
                                                      nrhs,
                                                      (float*)const_cast<void*>(A),
                                                      lda,
                                                      const_cast<int64_t*>(devIpiv),
                                                      (float*)B,
                                                      ldb));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeB == HIP_R_64F)
    {
        return rocblas2hip_status(rocsolver_dgetrs_64((rocblas_handle)handle,
                                                      hip2rocblas_operation(trans),
                                                      n,
                                                      nrhs,
                                                      (double*)const_cast<void*>(A),
                                                      lda,
                                                      const_cast<int64_t*>(devIpiv),
                                                      (double*)B,
                                                      ldb));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeB == HIP_C_32F)
    {
        return rocblas2hip_status(rocsolver_cgetrs_64((rocblas_handle)handle,
                                                      hip2rocblas_operation(trans),
                                                      n,
                                                      nrhs,
                                                      (rocblas_float_complex*)const_cast<void*>(A),
                                                      lda,
                                                      const_cast<int64_t*>(devIpiv),
                                                      (rocblas_float_complex*)B,
                                                      ldb));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeB == HIP_C_64F)
    {
        return rocblas2hip_status(rocsolver_zgetrs_64((rocblas_handle)handle,
                                                      hip2rocblas_operation(trans),
                                                      n,
                                                      nrhs,
                                                      (rocblas_double_complex*)const_cast<void*>(A),
                                                      lda,
                                                      const_cast<int64_t*>(devIpiv),
                                                      (rocblas_double_complex*)B,
                                                      ldb));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
}
catch(...)
{
    return exception2hip_status();
}

} //extern C
