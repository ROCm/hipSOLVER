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

inline constexpr auto hipsolver2string_status(hipsolverStatus_t value)
{
    switch(value)
    {
    case HIPSOLVER_STATUS_SUCCESS:
        return "HIPSOLVER_STATUS_SUCCESS";
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        return "HIPSOLVER_STATUS_NOT_INITIALIZED";
    case HIPSOLVER_STATUS_ALLOC_FAILED:
        return "HIPSOLVER_STATUS_ALLOC_FAILED";
    case HIPSOLVER_STATUS_INVALID_VALUE:
        return "HIPSOLVER_STATUS_INVALID_VALUE";
    case HIPSOLVER_STATUS_MAPPING_ERROR:
        return "HIPSOLVER_STATUS_MAPPING_ERROR";
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        return "HIPSOLVER_STATUS_EXECUTION_FAILED";
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        return "HIPSOLVER_STATUS_INTERNAL_ERROR";
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        return "HIPSOLVER_STATUS_NOT_SUPPORTED";
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        return "HIPSOLVER_STATUS_ARCH_MISMATCH";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR";
    }
    return "invalid";
}

char hipsolver2char_operation(hipsolverOperation_t value);

char hipsolver2char_fill(hipsolverFillMode_t value);

/* ============================================================================================ */
/*  Convert lapack char constants to hipsolver type. */

hipsolverStatus_t string2hipsolver_status(const std::string& value);

hipsolverOperation_t char2hipsolver_operation(char value);

hipsolverFillMode_t char2hipsolver_fill(char value);

#endif
