/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************/

#pragma once

#include "hipsolver.h"
#include "hipsolver_datatype2string.hpp"

template <typename T>
void cblas_getrf(int m, int n, T* A, int lda, int* ipiv, int* info);
