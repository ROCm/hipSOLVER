/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>

#include "hipsolver.h"
#include "utility.hpp"

hipsolver_rng_t hipsolver_rng(69069);
hipsolver_rng_t hipsolver_seed(hipsolver_rng);

template <>
char type2char<float>()
{
    return 's';
}

template <>
char type2char<double>()
{
    return 'd';
}

template <>
char type2char<hipsolverComplex>()
{
    return 'c';
}

template <>
char type2char<hipsolverDoubleComplex>()
{
    return 'z';
}

template <>
int type2int<float>(float val)
{
    return (int)val;
}

template <>
int type2int<double>(double val)
{
    return (int)val;
}

template <>
int type2int<hipsolverComplex>(hipsolverComplex val)
{
    return (int)val.real();
}

template <>
int type2int<hipsolverDoubleComplex>(hipsolverDoubleComplex val)
{
    return (int)val.real();
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  timing:*/

/* CPU Timer (in microseconds): no GPU synchronization
 */
double get_time_us_no_sync()
{
    namespace sc                         = std::chrono;
    const sc::steady_clock::time_point t = sc::steady_clock::now();
    return double(sc::duration_cast<sc::microseconds>(t.time_since_epoch()).count());
}

/* CPU Timer (in microseconds): synchronize with the default device and return wall time
 */
double get_time_us()
{
    hipDeviceSynchronize();
    return get_time_us_no_sync();
}

/* CPU Timer (in microseconds): synchronize with given queue/stream and return wall time
 */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    return get_time_us_no_sync();
}

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int query_device_property()
{
    int               device_count;
    hipsolverStatus_t count_status = (hipsolverStatus_t)hipGetDeviceCount(&device_count);
    if(count_status != HIPSOLVER_STATUS_SUCCESS)
    {
        printf("Query device error: cannot get device count \n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices \n", device_count);
    }

    for(int i = 0; i < device_count; i++)
    {
        hipDeviceProp_t   props;
        hipsolverStatus_t props_status = (hipsolverStatus_t)hipGetDeviceProperties(&props, i);
        if(props_status != HIPSOLVER_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s ------------------------------------------------------\n",
                   i,
                   props.name);
            printf("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem / 1e9,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf(
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);

            printf("-------------------------------------------------------------------------\n");
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int device_id)
{
    hipsolverStatus_t status = (hipsolverStatus_t)hipSetDevice(device_id);
    if(status != HIPSOLVER_STATUS_SUCCESS)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}

#ifdef __cplusplus
}
#endif
