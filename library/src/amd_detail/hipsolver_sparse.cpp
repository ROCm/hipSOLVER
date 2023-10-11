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

#ifdef HAVE_ROCSPARSE
#include "cholmod.h"
#include "rocsparse/rocsparse.h"
#endif

extern "C" {

/******************** HANDLE ********************/
struct hipsolverSpHandle
{
#ifdef HAVE_ROCSPARSE
    rocblas_handle   handle;
    rocsolver_rfinfo rfinfo;
    cholmod_common   c_handle;

    rocblas_int n, nnzA, batch_count;

    rocblas_int* dPtrA;
    rocblas_int* dIndA;
    double*      dValA;

    rocblas_int* dQ;

    char* d_buffer;

    // Constructor
    explicit hipsolverSpHandle()
        : n(0)
        , nnzA(0)
        , d_buffer(nullptr)
    {
    }

    // Allocate device memory
    hipsolverStatus_t malloc_device(int n, int nnzA)
    {
        if(n < 0 || nnzA < 0)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        if(this->n < n || this->nnzA < nnzA)
        {
            if(this->d_buffer)
            {
                if(hipFree(this->d_buffer) != hipSuccess)
                    return HIPSOLVER_STATUS_INTERNAL_ERROR;
                this->d_buffer = nullptr;
            }

            size_t size_dPtrA = sizeof(rocblas_int) * (n + 1);
            size_t size_dIndA = sizeof(rocblas_int) * nnzA;
            size_t size_dValA = sizeof(double) * nnzA;

            size_t size_dQ = sizeof(rocblas_int) * n;

            // 128 byte alignment
            size_dPtrA = ((size_dPtrA - 1) / 128 + 1) * 128;
            size_dIndA = ((size_dIndA - 1) / 128 + 1) * 128;
            size_dValA = ((size_dValA - 1) / 128 + 1) * 128;
            size_dQ    = ((size_dQ - 1) / 128 + 1) * 128;

            size_t size_buffer = size_dPtrA + size_dIndA + size_dValA + size_dQ;

            if(hipMalloc(&this->d_buffer, size_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_ALLOC_FAILED;

            char* temp_buf;
            this->dPtrA = (rocblas_int*)(temp_buf = this->d_buffer);

            this->dIndA = (rocblas_int*)(temp_buf += size_dPtrA);

            this->dQ = (rocblas_int*)(temp_buf += size_dIndA);

            this->dValA = (double*)(temp_buf += size_dQ);

            this->n    = n;
            this->nnzA = nnzA;
        }

        return HIPSOLVER_STATUS_SUCCESS;
    }

    // Free memory
    void free_all()
    {
        if(this->d_buffer)
        {
            hipFree(this->d_buffer);
            this->d_buffer = nullptr;
        }
    }

    // Convert base one indices to base zero, and copy float values into double array
    void prep_input(rocsparse_index_base indbase,
                    int                  n,
                    int                  nnz,
                    int*                 ptr,
                    int*                 ind,
                    double*              val,
                    float*               src_val)
    {
        int count;
        if(indbase == rocsparse_index_base_one)
        {
            for(int i = 0; i <= n; i++)
                ptr[i] -= 1;
            count = std::min(nnz, ptr[n]);
            for(int i = 0; i < count; i++)
                ind[i] -= 1;
        }
        else
            count = std::min(nnz, ptr[n]);

        if(src_val)
        {
            for(int i = 0; i < count; i++)
                val[i] = (double)src_val[i];
        }
    }
    // Copy float values into double array
    void prep_input(int n, double* val, float* src_val)
    {
        if(src_val)
        {
            for(int i = 0; i < n; i++)
                val[i] = (double)src_val[i];
        }
    }

    // Convert base zero indices to base one, and copy double values into float array
    void prep_output(rocsparse_index_base indbase,
                     int                  n,
                     int                  nnz,
                     int*                 ptr,
                     int*                 ind,
                     double*              val,
                     float*               dest_val)
    {
        int count;
        if(indbase == rocsparse_index_base_one)
        {
            count = std::min(nnz, ptr[n]);
            for(int i = 0; i <= n; i++)
                ptr[i] += 1;
            for(int i = 0; i < count; i++)
                ind[i] += 1;
        }
        else
            count = std::min(nnz, ptr[n]);

        if(dest_val)
        {
            for(int i = 0; i < count; i++)
                dest_val[i] = (float)val[i];
        }
    }
    // Copy double values into float array
    void prep_output(int n, double* val, float* dest_val)
    {
        if(dest_val)
        {
            for(int i = 0; i < n; i++)
                dest_val[i] = (float)val[i];
        }
    }
#endif
};

hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle)
try
{
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = new hipsolverSpHandle;
    rocblas_status     status;

    if((status = rocblas_create_handle(&sp->handle)) != rocblas_status_success)
    {
        delete sp;
        return rocblas2hip_status(status);
    }

    if((status = rocsolver_create_rfinfo(&sp->rfinfo, sp->handle)) != rocblas_status_success)
    {
        rocblas_destroy_handle(sp->handle);
        delete sp;
        return rocblas2hip_status(status);
    }

    int ret;
    if((ret = cholmod_start(&sp->c_handle)) != true)
    {
        rocblas_destroy_handle(sp->handle);
        rocsolver_destroy_rfinfo(sp->rfinfo);
        delete sp;
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    *handle = sp;
    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle)
try
{
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    sp->free_all();
    rocblas_destroy_handle(sp->handle);
    rocsolver_destroy_rfinfo(sp->rfinfo);
    cholmod_finish(&sp->c_handle);
    delete sp;

    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId)
try
{
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    return rocblas2hip_status(rocblas_set_stream(sp->handle, streamId));
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
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
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
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
    float*          sngVal = (float*)malloc(sizeof(float) * nnzA);
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(sngVal, csrVal, sizeof(float) * nnzA, hipMemcpyDeviceToHost));
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, sngVal);
    free(sngVal);

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

    // set up T
    cholmod_sparse* c_TT = cholmod_factor_to_sparse(c_L, &sp->c_handle);
    cholmod_sparse* c_T  = cholmod_transpose(c_TT, 2, &sp->c_handle);
    CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, c_T->nzmax));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dPtrA, c_T->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dIndA, c_T->i, sizeof(rocblas_int) * c_T->nzmax, hipMemcpyHostToDevice));

    sngVal = (float*)malloc(sizeof(float) * c_T->nzmax);
    sp->prep_output(c_T->nzmax, (double*)c_T->x, sngVal);
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dValA, sngVal, sizeof(float) * c_T->nzmax, hipMemcpyHostToDevice));
    free(sngVal);

    // set up Q
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dQ, c_L->Perm, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(float) * n, hipMemcpyDeviceToDevice));

    // solve for x
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(sp->rfinfo, rocsolver_rfinfo_mode_cholesky));
    CHECK_ROCBLAS_ERROR(rocsolver_scsrrf_analysis(sp->handle,
                                                  n,
                                                  1,
                                                  nnzA,
                                                  (int*)csrRowPtr,
                                                  (int*)csrColInd,
                                                  (float*)csrVal,
                                                  c_T->nzmax,
                                                  sp->dPtrA,
                                                  sp->dIndA,
                                                  (float*)sp->dValA,
                                                  nullptr,
                                                  sp->dQ,
                                                  x,
                                                  n,
                                                  sp->rfinfo));
    CHECK_ROCBLAS_ERROR(rocsolver_scsrrf_solve(sp->handle,
                                               n,
                                               1,
                                               c_T->nzmax,
                                               sp->dPtrA,
                                               sp->dIndA,
                                               (float*)sp->dValA,
                                               nullptr,
                                               sp->dQ,
                                               x,
                                               n,
                                               sp->rfinfo));

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_sparse(&c_TT, &sp->c_handle);
    cholmod_free_sparse(&c_T, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
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
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
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
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(c_A->x, csrVal, sizeof(double) * nnzA, hipMemcpyDeviceToHost));
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, nullptr);

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

    // set up T
    cholmod_sparse* c_TT = cholmod_factor_to_sparse(c_L, &sp->c_handle);
    cholmod_sparse* c_T  = cholmod_transpose(c_TT, 2, &sp->c_handle);
    CHECK_HIPSOLVER_ERROR(sp->malloc_device(n, c_T->nzmax));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dPtrA, c_T->p, sizeof(rocblas_int) * (n + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        (void*)sp->dIndA, c_T->i, sizeof(rocblas_int) * c_T->nzmax, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dValA, c_T->x, sizeof(double) * c_T->nzmax, hipMemcpyHostToDevice));

    // set up Q
    CHECK_HIP_ERROR(
        hipMemcpy((void*)sp->dQ, c_L->Perm, sizeof(rocblas_int) * n, hipMemcpyHostToDevice));

    // set up B
    CHECK_HIP_ERROR(hipMemcpy((void*)x, b, sizeof(double) * n, hipMemcpyDeviceToDevice));

    // solve for x
    CHECK_ROCBLAS_ERROR(rocsolver_set_rfinfo_mode(sp->rfinfo, rocsolver_rfinfo_mode_cholesky));
    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_analysis(sp->handle,
                                                  n,
                                                  1,
                                                  nnzA,
                                                  (int*)csrRowPtr,
                                                  (int*)csrColInd,
                                                  (double*)csrVal,
                                                  c_T->nzmax,
                                                  sp->dPtrA,
                                                  sp->dIndA,
                                                  sp->dValA,
                                                  nullptr,
                                                  sp->dQ,
                                                  x,
                                                  n,
                                                  sp->rfinfo));
    CHECK_ROCBLAS_ERROR(rocsolver_dcsrrf_solve(sp->handle,
                                               n,
                                               1,
                                               c_T->nzmax,
                                               sp->dPtrA,
                                               sp->dIndA,
                                               sp->dValA,
                                               nullptr,
                                               sp->dQ,
                                               x,
                                               n,
                                               sp->rfinfo));

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_sparse(&c_TT, &sp->c_handle);
    cholmod_free_sparse(&c_T, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
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
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
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
    float*          sngVal = (float*)malloc(sizeof(float) * nnzA);
    cholmod_sparse* c_A
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1));
    memcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA);
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, (float*)csrVal);
    free(sngVal);

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

    // set up B
    cholmod_dense* c_b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &sp->c_handle);
    sp->prep_input(n, (double*)c_b->x, (float*)b);

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, c_b, &sp->c_handle);

    // copy back results
    sp->prep_output(n, (double*)c_x->x, (float*)x);

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_b, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
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
#ifdef HAVE_ROCSPARSE
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(n < 0 || nnzA < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!csrRowPtr || !csrColInd || !csrVal || !descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!b || !x || !singularity)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(reorder < 0 || reorder > 3)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    rocsparse_matrix_type mattype = rocsparse_get_mat_type((rocsparse_mat_descr)descrA);
    // rocsparse_fill_mode fillmode = rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA);
    // rocsparse_diag_type diagtype = rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA);
    rocsparse_index_base indbase = rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA);
    if(mattype != rocsparse_matrix_type_general)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if(indbase != rocsparse_index_base_zero && indbase != rocsparse_index_base_one)
        return HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    hipsolverSpHandle* sp = (hipsolverSpHandle*)handle;
    *singularity          = -1;

    sp->c_handle.nmethods  = 1; // use 1 reordering method
    sp->c_handle.postorder = false; // no postordering
    sp->c_handle.final_ll  = true; // factorize as LL' not LDL'
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
        = cholmod_allocate_sparse(n, n, nnzA, true, true, 1, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_A->p, csrRowPtr, sizeof(rocblas_int) * (n + 1));
    memcpy(c_A->i, csrColInd, sizeof(rocblas_int) * nnzA);
    memcpy(c_A->x, csrVal, sizeof(double) * nnzA);
    sp->prep_input(indbase, n, nnzA, (int*)c_A->p, (int*)c_A->i, (double*)c_A->x, nullptr);

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

    // set up B
    cholmod_dense* c_b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &sp->c_handle);
    memcpy(c_b->x, b, sizeof(double) * n);

    // solve for x
    cholmod_dense* c_x = cholmod_solve(CHOLMOD_A, c_L, c_b, &sp->c_handle);

    // copy back results
    memcpy((void*)x, c_x->x, sizeof(double) * n);

    // free resources
    cholmod_free_sparse(&c_A, &sp->c_handle);
    cholmod_free_factor(&c_L, &sp->c_handle);
    cholmod_free_dense(&c_b, &sp->c_handle);
    cholmod_free_dense(&c_x, &sp->c_handle);

    return HIPSOLVER_STATUS_SUCCESS;
#else
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
#endif
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
