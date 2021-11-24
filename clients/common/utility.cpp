/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <chrono>

#include "hipsolver/hipsolver.h"
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
    hipsolverStatus_t status = (hipsolverStatus_t)hipGetDeviceCount(&device_count);
    if(status != HIPSOLVER_STATUS_SUCCESS)
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
        hipsolverStatus_t status = (hipsolverStatus_t)hipGetDeviceProperties(&props, i);
        if(status != HIPSOLVER_STATUS_SUCCESS)
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
