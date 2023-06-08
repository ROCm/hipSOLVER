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

#pragma once

#include "hipsolver.h"
#include <cusolverDn.h>
#include <cusolverRf.h>

struct hipsolverRfHandle
{
    cusolverRfHandle_t handle;

    // Constructor
    hipsolverRfHandle();

    hipsolverRfHandle(const hipsolverRfHandle&) = delete;

    hipsolverRfHandle(hipsolverRfHandle&&) = delete;

    hipsolverRfHandle& operator=(const hipsolverRfHandle&) = delete;

    hipsolverRfHandle& operator=(hipsolverRfHandle&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup();
    hipsolverStatus_t teardown();
};

struct hipsolverGesvdjInfo
{
    gesvdjInfo_t info;

    // Constructor
    hipsolverGesvdjInfo();

    hipsolverGesvdjInfo(const hipsolverGesvdjInfo&) = delete;

    hipsolverGesvdjInfo(hipsolverGesvdjInfo&&) = delete;

    hipsolverGesvdjInfo& operator=(const hipsolverGesvdjInfo&) = delete;

    hipsolverGesvdjInfo& operator=(hipsolverGesvdjInfo&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup();
    hipsolverStatus_t teardown();
};

struct hipsolverSyevjInfo
{
    syevjInfo_t info;

    // Constructor
    hipsolverSyevjInfo();

    hipsolverSyevjInfo(const hipsolverSyevjInfo&) = delete;

    hipsolverSyevjInfo(hipsolverSyevjInfo&&) = delete;

    hipsolverSyevjInfo& operator=(const hipsolverSyevjInfo&) = delete;

    hipsolverSyevjInfo& operator=(hipsolverSyevjInfo&&) = delete;

    // Allocate resources
    hipsolverStatus_t setup();
    hipsolverStatus_t teardown();
};
