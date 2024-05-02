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

#include "lib_macros.hpp"
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
    return hipsolver::exception2hip_status();
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
    return hipsolver::exception2hip_status();
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
    return hipsolver::exception2hip_status();
}

} //extern C
