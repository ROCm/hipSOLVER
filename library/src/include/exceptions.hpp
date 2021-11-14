/* ************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver/hipsolver.h"
#include <exception>

// Convert the current C++ exception to hipsolverStatus_t
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent hipsolverStatus_t here
inline hipsolverStatus_t exception2hip_status(std::exception_ptr e = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return HIPSOLVER_STATUS_SUCCESS;
}
catch(const hipsolverStatus_t& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return HIPSOLVER_STATUS_ALLOC_FAILED;
}
catch(...)
{
    return HIPSOLVER_STATUS_INTERNAL_ERROR;
}
