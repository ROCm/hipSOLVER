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

#include "cholmod.h"
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

    cholmod_common c_handle;
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

    int ret;
    if((ret = cholmod_start(&sp->c_handle)) != true)
    {
        rocblas_destroy_handle(sp->handle);
        delete sp;
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
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
    cholmod_finish(&sp->c_handle);
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, -1, CHOLMOD_REAL, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));

    int count_A = std::min(nnzA, ((int*)c_A->p)[n]);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * count_A, hipMemcpyDeviceToHost));

    float* sngVal = (float*)malloc(sizeof(float) * nnzA);
    CHECK_HIP_ERROR(hipMemcpy(sngVal, csrVal, sizeof(float) * count_A, hipMemcpyDeviceToHost));

    double* dblVal = static_cast<double*>(c_A->x);
    for(int i = 0; i < count_A; i++)
        dblVal[i] = sngVal[i];

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(c_A, &sp->c_handle);
    int             status = cholmod_factorize(c_A, c_L, &sp->c_handle);
    if(status != true)
    {
        free(sngVal);
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        free(sngVal);
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // copy back results
    count_A = std::min(nnzA, ((int*)c_L->p)[n]);
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrRowPtr, c_L->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrColInd, c_L->i, sizeof(rocblas_int) * count_A, hipMemcpyHostToDevice));

    dblVal = static_cast<double*>(c_L->x);
    for(int i = 0; i < count_A; i++)
        sngVal[i] = (float)dblVal[i];
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrVal, sngVal, sizeof(float) * count_A, hipMemcpyHostToDevice));

    // free resources
    free(sngVal);
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    // TODO: Call solve on GPU

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, -1, CHOLMOD_REAL, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));

    int count_A = std::min(nnzA, ((int*)c_A->p)[n]);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * count_A, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(c_A->x, csrVal, sizeof(double) * count_A, hipMemcpyDeviceToHost));

    if(tolerance > 0)
        cholmod_drop(tolerance, c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(c_A, &sp->c_handle);
    int             status = cholmod_factorize(c_A, c_L, &sp->c_handle);
    if(status != true)
    {
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        cholmod_free_sparse(&c_A, &sp->c_handle);
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // copy back results
    count_A = std::min(nnzA, ((int*)c_L->p)[n]);
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrRowPtr, c_L->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrColInd, c_L->i, sizeof(rocblas_int) * count_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)csrVal, c_L->x, sizeof(double) * count_A, hipMemcpyHostToDevice));

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    // TODO: Call solve on GPU

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods = 1;
    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    int count_A = std::min(nnzA, csrRowPtr[n]);

    double* dblVal = (double*)malloc(sizeof(double) * nnzA);
    for(int i = 0; i < count_A; i++)
        dblVal[i] = csrVal[i];

    cholmod_sparse c_A;
    c_A.nrow = c_A.ncol = n;
    c_A.nzmax           = nnzA;
    c_A.p               = (void*)csrRowPtr;
    c_A.i               = (void*)csrColInd;
    c_A.x               = (void*)dblVal;
    c_A.stype           = -1;
    c_A.itype           = CHOLMOD_INT;
    c_A.xtype           = CHOLMOD_REAL;
    c_A.dtype           = CHOLMOD_DOUBLE;
    c_A.packed          = true;

    if(tolerance > 0)
        cholmod_drop(tolerance, &c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(&c_A, &sp->c_handle);
    int             status = cholmod_factorize(&c_A, c_L, &sp->c_handle);
    free(dblVal);
    if(status != true)
    {
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // set up B
    double* dblB = (double*)malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)
        dblB[i] = b[i];

    cholmod_dense c_b;
    c_b.nrow = c_b.nzmax = c_b.d = n;
    c_b.ncol                     = 1;
    c_b.x                        = (void*)dblB;
    c_b.xtype                    = CHOLMOD_REAL;
    c_b.dtype                    = CHOLMOD_DOUBLE;

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, &c_b, &sp->c_handle);
    free(dblB);

    // copy back results
    count_A = std::min(nnzA, ((int*)c_L->p)[n]);
    memcpy((void*)csrRowPtr, c_L->p, sizeof(int) * (n + 1));
    memcpy((void*)csrColInd, c_L->i, sizeof(int) * count_A);

    dblVal = static_cast<double*>(c_L->x);
    dblB   = static_cast<double*>(c_x->x);
    for(int i = 0; i < count_A; i++)
        ((float*)csrVal)[i] = (float)dblVal[i];
    for(int i = 0; i < n; i++)
        x[i] = (float)dblB[i];

    // free resources
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
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
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    switch(reorder)
    {
    case 1:
    case 2:
        sp->c_handle.method[0].ordering = CHOLMOD_AMD;
        break;
    case 3:
        sp->c_handle.method[0].ordering = CHOLMOD_METIS;
        break;
    default:
        sp->c_handle.method[0].ordering = CHOLMOD_NATURAL;
    }

    // set up A
    int            count_A = std::min(nnzA, csrRowPtr[n]);
    cholmod_sparse c_A;
    c_A.nrow = c_A.ncol = n;
    c_A.nzmax           = nnzA;
    c_A.p               = (void*)csrRowPtr;
    c_A.i               = (void*)csrColInd;
    c_A.x               = (void*)csrVal;
    c_A.stype           = -1;
    c_A.itype           = CHOLMOD_INT;
    c_A.xtype           = CHOLMOD_REAL;
    c_A.dtype           = CHOLMOD_DOUBLE;
    c_A.packed          = true;

    if(tolerance > 0)
        cholmod_drop(tolerance, &c_A, &sp->c_handle);

    // factorize A
    cholmod_factor* c_L    = cholmod_analyze(&c_A, &sp->c_handle);
    int             status = cholmod_factorize(&c_A, c_L, &sp->c_handle);
    if(status != true)
    {
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }
    if(sp->c_handle.status == CHOLMOD_NOT_POSDEF)
    {
        *singularity = c_L->minor;
        cholmod_free_factor(&c_L, &sp->c_handle);
        return HIPSOLVER_STATUS_SUCCESS;
    }

    // set up B
    cholmod_dense c_b;
    c_b.nrow = c_b.nzmax = c_b.d = n;
    c_b.ncol                     = 1;
    c_b.x                        = (void*)b;
    c_b.xtype                    = CHOLMOD_REAL;
    c_b.dtype                    = CHOLMOD_DOUBLE;

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, &c_b, &sp->c_handle);

    // copy back results
    count_A = std::min(nnzA, ((int*)c_L->p)[n]);
    memcpy((void*)csrRowPtr, c_L->p, sizeof(int) * (n + 1));
    memcpy((void*)csrColInd, c_L->i, sizeof(int) * count_A);
    memcpy((void*)csrVal, c_L->x, sizeof(double) * count_A);
    memcpy((void*)x, c_x->x, sizeof(double) * n);

    // free resources
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
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
