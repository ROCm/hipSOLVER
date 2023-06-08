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

#include "hipsolver_types.hpp"
#include "hipsolver_conversions.hpp"

/******************** RF HANDLE ********************/
hipsolverRfHandle::hipsolverRfHandle()
    : fast_mode(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
    , matrix_format(HIPSOLVERRF_MATRIX_FORMAT_CSR)
    , diag_format(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)
    , numeric_boost(HIPSOLVERRF_NUMERIC_BOOST_NOT_USED)
    , fact_alg(HIPSOLVERRF_FACTORIZATION_ALG0)
    , solve_alg(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)
    , n(0)
    , nnzA(0)
    , nnzL(0)
    , nnzU(0)
    , nnzLU(0)
    , batch_count(0)
    , effective_zero(0.0)
    , boost_val(0.0)
    , d_buffer(nullptr)
    , h_buffer(nullptr)
{
}

hipsolverStatus_t hipsolverRfHandle::setup()
{
    rocblas_status status;

    if((status = rocblas_create_handle(&this->handle)) != rocblas_status_success)
        return rocblas2hip_status(status);

    if((status = rocsolver_create_rfinfo(&this->rfinfo, this->handle)) != rocblas_status_success)
    {
        rocblas_destroy_handle(this->handle);
        return rocblas2hip_status(status);
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverRfHandle::teardown()
{
    rocblas_status status;

    if(free_mem() != HIPSOLVER_STATUS_SUCCESS)
    {
        rocsolver_destroy_rfinfo(this->rfinfo);
        rocblas_destroy_handle(this->handle);
        return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    if((status = rocsolver_destroy_rfinfo(this->rfinfo)) != rocblas_status_success)
    {
        rocblas_destroy_handle(this->handle);
        return rocblas2hip_status(status);
    }

    if((status = rocblas_destroy_handle(this->handle)) != rocblas_status_success)
        return rocblas2hip_status(status);

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverRfHandle::malloc_device(int n, int nnzA, int nnzL, int nnzU)
{
    if(n < 0 || nnzA < 0 || nnzL < 0 || nnzU < 0)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    if(this->n != n || this->nnzA != nnzA || this->nnzL != nnzL || this->nnzU != nnzU)
    {
        int nnzLU = nnzL - n + nnzU;

        if(free_mem() != HIPSOLVER_STATUS_SUCCESS)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;

        size_t size_dPtrA = sizeof(rocblas_int) * (n + 1);
        size_t size_dIndA = sizeof(rocblas_int) * nnzA;
        size_t size_dValA = sizeof(double) * nnzA;

        size_t size_dPtrL = sizeof(rocblas_int) * (n + 1);
        size_t size_dIndL = sizeof(rocblas_int) * nnzL;
        size_t size_dValL = sizeof(double) * nnzL;

        size_t size_dPtrU = sizeof(rocblas_int) * (n + 1);
        size_t size_dIndU = sizeof(rocblas_int) * nnzU;
        size_t size_dValU = sizeof(double) * nnzU;

        size_t size_dPtrLU = sizeof(rocblas_int) * (n + 1);
        size_t size_dIndLU = sizeof(rocblas_int) * nnzLU;
        size_t size_dValLU = sizeof(double) * nnzLU;

        size_t size_dP = sizeof(rocblas_int) * n;
        size_t size_dQ = sizeof(rocblas_int) * n;

        // 128 byte alignment
        size_dPtrL  = ((size_dPtrL - 1) / 128 + 1) * 128;
        size_dIndL  = ((size_dIndL - 1) / 128 + 1) * 128;
        size_dValL  = ((size_dValL - 1) / 128 + 1) * 128;
        size_dPtrU  = ((size_dPtrU - 1) / 128 + 1) * 128;
        size_dIndU  = ((size_dIndU - 1) / 128 + 1) * 128;
        size_dValU  = ((size_dValU - 1) / 128 + 1) * 128;
        size_dPtrLU = ((size_dPtrLU - 1) / 128 + 1) * 128;
        size_dIndLU = ((size_dIndLU - 1) / 128 + 1) * 128;
        size_dValLU = ((size_dValLU - 1) / 128 + 1) * 128;
        size_dP     = ((size_dP - 1) / 128 + 1) * 128;
        size_dQ     = ((size_dQ - 1) / 128 + 1) * 128;

        size_t size_buffer = size_dPtrA + size_dIndA + size_dValA + size_dPtrL + size_dIndL
                             + size_dValL + size_dPtrU + size_dIndU + size_dValU + size_dPtrLU
                             + size_dIndLU + size_dValLU + size_dP + size_dQ;

        if(hipMalloc(&this->d_buffer, size_buffer) != hipSuccess)
            return HIPSOLVER_STATUS_ALLOC_FAILED;

        char* temp_buf;
        this->dPtrA  = (rocblas_int*)(temp_buf = this->d_buffer);
        this->dPtrL  = (rocblas_int*)(temp_buf += size_dPtrA);
        this->dPtrU  = (rocblas_int*)(temp_buf += size_dPtrL);
        this->dPtrLU = (rocblas_int*)(temp_buf += size_dPtrU);

        this->dIndA  = (rocblas_int*)(temp_buf += size_dPtrLU);
        this->dIndL  = (rocblas_int*)(temp_buf += size_dIndA);
        this->dIndU  = (rocblas_int*)(temp_buf += size_dIndL);
        this->dIndLU = (rocblas_int*)(temp_buf += size_dIndU);

        this->dP = (rocblas_int*)(temp_buf += size_dIndLU);
        this->dQ = (rocblas_int*)(temp_buf += size_dP);

        this->dValA  = (double*)(temp_buf += size_dQ);
        this->dValL  = (double*)(temp_buf += size_dValA);
        this->dValU  = (double*)(temp_buf += size_dValL);
        this->dValLU = (double*)(temp_buf += size_dValU);

        this->n     = n;
        this->nnzA  = nnzA;
        this->nnzL  = nnzL;
        this->nnzU  = nnzU;
        this->nnzLU = nnzLU;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverRfHandle::malloc_host()
{
    if(!this->h_buffer)
    {
        size_t size_hPtrL = sizeof(rocblas_int) * (n + 1);
        size_t size_hIndL = sizeof(rocblas_int) * nnzL;
        size_t size_hValL = sizeof(double) * nnzL;

        size_t size_hPtrU = sizeof(rocblas_int) * (n + 1);
        size_t size_hIndU = sizeof(rocblas_int) * nnzU;
        size_t size_hValU = sizeof(double) * nnzU;

        size_t size_hPtrLU = sizeof(rocblas_int) * (n + 1);
        size_t size_hIndLU = sizeof(rocblas_int) * nnzLU;
        size_t size_hValLU = sizeof(double) * nnzLU;

        // 128 byte alignment
        size_hPtrL  = ((size_hPtrL - 1) / 128 + 1) * 128;
        size_hIndL  = ((size_hIndL - 1) / 128 + 1) * 128;
        size_hValL  = ((size_hValL - 1) / 128 + 1) * 128;
        size_hPtrU  = ((size_hPtrU - 1) / 128 + 1) * 128;
        size_hIndU  = ((size_hIndU - 1) / 128 + 1) * 128;
        size_hValU  = ((size_hValU - 1) / 128 + 1) * 128;
        size_hPtrLU = ((size_hPtrLU - 1) / 128 + 1) * 128;
        size_hIndLU = ((size_hIndLU - 1) / 128 + 1) * 128;
        size_hValLU = ((size_hValLU - 1) / 128 + 1) * 128;

        size_t size_buffer = size_hPtrL + size_hIndL + size_hValL + size_hPtrU + size_hIndU
                             + size_hValU + size_hPtrLU + size_hIndLU + size_hValLU;

        this->h_buffer = (char*)malloc(size_buffer);
        if(!this->h_buffer)
            return HIPSOLVER_STATUS_ALLOC_FAILED;

        char* temp_buf;
        this->hPtrL  = (rocblas_int*)(temp_buf = this->h_buffer);
        this->hPtrU  = (rocblas_int*)(temp_buf += size_hPtrL);
        this->hPtrLU = (rocblas_int*)(temp_buf += size_hPtrU);

        this->hIndL  = (rocblas_int*)(temp_buf += size_hPtrLU);
        this->hIndU  = (rocblas_int*)(temp_buf += size_hIndL);
        this->hIndLU = (rocblas_int*)(temp_buf += size_hIndU);

        this->hValL  = (double*)(temp_buf += size_hIndLU);
        this->hValU  = (double*)(temp_buf += size_hValL);
        this->hValLU = (double*)(temp_buf += size_hValU);
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverRfHandle::free_mem()
{
    if(this->h_buffer)
    {
        try
        {
            free(this->h_buffer);
        }
        catch(...)
        {
            if(this->d_buffer)
                hipFree(this->d_buffer);
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
        }
        this->h_buffer = nullptr;
    }

    if(this->d_buffer)
    {
        if(hipFree(this->d_buffer) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
        this->d_buffer = nullptr;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

/******************** GESVDJ PARAMS ********************/
hipsolverGesvdjInfo::hipsolverGesvdjInfo()
    : capacity(0)
    , batch_count(0)
    , n_sweeps(nullptr)
    , residual(nullptr)
    , max_sweeps(100)
    , tolerance(0)
    , is_batched(false)
    , is_float(false)
    , sort_eig(true)
    , d_buffer(nullptr)
{
}

hipsolverStatus_t hipsolverGesvdjInfo::setup(int bc)
{
    if(capacity < bc)
    {
        if(capacity > 0)
        {
            if(hipFree(d_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_INTERNAL_ERROR;
        }

        if(hipMalloc(&d_buffer, sizeof(int) * bc + sizeof(double) * bc) != hipSuccess)
        {
            capacity = 0;
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        }

        n_sweeps    = (int*)d_buffer;
        residual    = (double*)(n_sweeps + bc);
        capacity    = bc;
        batch_count = bc;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverGesvdjInfo::teardown()
{
    if(capacity > 0)
    {
        capacity = 0;
        if(hipFree(d_buffer) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

/******************** SYEVJ PARAMS ********************/
hipsolverSyevjInfo::hipsolverSyevjInfo()
    : capacity(0)
    , batch_count(0)
    , n_sweeps(nullptr)
    , residual(nullptr)
    , max_sweeps(100)
    , tolerance(0)
    , is_batched(false)
    , is_float(false)
    , sort_eig(true)
    , d_buffer(nullptr)
{
}

hipsolverStatus_t hipsolverSyevjInfo::setup(int bc)
{
    if(capacity < bc)
    {
        if(capacity > 0)
        {
            if(hipFree(d_buffer) != hipSuccess)
                return HIPSOLVER_STATUS_INTERNAL_ERROR;
        }

        if(hipMalloc(&d_buffer, (sizeof(int) + sizeof(double)) * bc) != hipSuccess)
        {
            capacity = 0;
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        }

        n_sweeps    = (int*)d_buffer;
        residual    = (double*)(n_sweeps + bc);
        capacity    = bc;
        batch_count = bc;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverSyevjInfo::teardown()
{
    if(capacity > 0)
    {
        capacity = 0;
        if(hipFree(d_buffer) != hipSuccess)
            return HIPSOLVER_STATUS_INTERNAL_ERROR;
    }

    return HIPSOLVER_STATUS_SUCCESS;
}
