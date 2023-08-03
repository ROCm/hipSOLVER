/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief Implementation of the compatibility sparse APIs that require especial
 *  calls to hipSOLVER or rocSOLVER.
 */

#include "error_macros.hpp"
#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

extern "C" {

/******************** HANDLE ********************/
struct hipsolverSpHandle
{
    rocblas_handle handle;
};

hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = new hipsolverSpHandle;
    rocblas_status     status;

    if((status = rocblas_create_handle(&sp->handle)) != rocblas_status_success)
    {
        delete sp;
        return rocblas2hip_status(status);
    }

    *handle = sp;
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    rocblas_destroy_handle(sp->handle);
    delete sp;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    return rocblas2hip_status(rocblas_set_stream(sp->handle, streamId));
}
catch(...)
{
    return exception2hip_status();
}

/******************** CSRLSVCHOL ********************/
hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t       handle,
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
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t       handle,
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
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvchol(hipsolverSpHandle_t       handle,
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
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvchol(hipsolverSpHandle_t       handle,
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
                                         int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}*/

hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t       handle,
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
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t       handle,
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
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvcholHost(hipsolverSpHandle_t       handle,
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
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvcholHost(hipsolverSpHandle_t       handle,
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
                                             int*                      singularity)
try
{
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
}
catch(...)
{
    return exception2hip_status();
}*/

} //extern C
