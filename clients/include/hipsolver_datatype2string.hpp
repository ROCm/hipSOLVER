/* ************************************************************************
 * Copyright 2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#ifndef HIPSOLVER_DATATYPE2STRING_H_
#define HIPSOLVER_DATATYPE2STRING_H_

#include "hipsolver.h"
#include <ostream>
#include <string>

// Complex output
inline std::ostream& operator<<(std::ostream& os, const hipsolverComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const hipsolverDoubleComplex& x)
{
    os << "'(" << x.real() << "," << x.imag() << ")'";
    return os;
}

/* ============================================================================================ */
/*  Convert hipsolver constants to lapack char. */

char hipsolver2char_operation(hipsolverOperation_t value);

char hipsolver2char_fill(hipsolverFillMode_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipsolver type. */

hipsolverOperation_t char2hipsolver_operation(char value);

hipsolverFillMode_t char2hipsolver_fill(char value);

#endif
