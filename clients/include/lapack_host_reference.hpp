/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once
#ifndef LAPACK_HOST_REFERENCE
#define LAPACK_HOST_REFERENCE

#include "hipsolver.h"
#include "hipsolver_datatype2string.hpp"

template <typename T>
void cblas_getrf(int m, int n, T* A, int lda, int* ipiv, int* info);

#endif /* LAPACK_HOST_REFERENCE */
