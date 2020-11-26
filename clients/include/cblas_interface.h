/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once
#ifndef CBLAS_INTERFACE
#define CBLAS_INTERFACE

#include "hipsolver.h"
#include "hipsolver_datatype2string.hpp"

template <typename T>
void cblas_getrf(int m, int n, T* A, int lda, int* ipiv, int* info);

#endif /* CBLAS_INTERFACE */
