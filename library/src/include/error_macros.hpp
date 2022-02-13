/* ************************************************************************
 * Copyright 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"

#define CHECK_HIPSOLVER_ERROR(STATUS)           \
    do                                          \
    {                                           \
        hipsolverStatus_t _status = (STATUS);   \
        if(_status != HIPSOLVER_STATUS_SUCCESS) \
            return _status;                     \
    } while(0)

#define CHECK_ROCBLAS_ERROR(STATUS)             \
    do                                          \
    {                                           \
        rocblas_status _status = (STATUS);      \
        if(_status != rocblas_status_success)   \
            return rocblas2hip_status(_status); \
    } while(0)
