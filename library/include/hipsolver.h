/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well.
//!
//! This is the master include file for hipsolver, wrapping around rocsolver and cusolver
//
#ifndef HIPSOLVER_H
#define HIPSOLVER_H

#include "hipsolver-export.h"
#include "hipsolver-version.h"
#include <hip/hip_runtime_api.h>
#include <stdint.h>

#endif
