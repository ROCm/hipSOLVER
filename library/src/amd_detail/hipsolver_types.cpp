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
