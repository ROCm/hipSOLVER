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
 *  to hipSOLVER on the cuSOLVER side.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"

#include <cusolverDn.h>

extern "C" {

hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* params)
try
{
    return cuda2hip_status(cusolverDnCreateParams((cusolverDnParams_t*)params));
}
catch(...)
{
    return exception2hip_status();
}

hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params)
try
{
    return cuda2hip_status(cusolverDnDestroyParams((cusolverDnParams_t)params));
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
    return cuda2hip_status(cusolverDnSetAdvOptions(
        (cusolverDnParams_t)params, hip2cuda_function(func), hip2cuda_algmode(alg)));
}
catch(...)
{
    return exception2hip_status();
}

} //extern C
