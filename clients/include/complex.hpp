/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"
#include <complex>

typedef struct hipsolverComplex
{
private:
    float x, y;

public:
    hipsolverComplex() = default;

    hipsolverComplex(float r, float i = 0)
        : x(r)
        , y(i)
    {
    }

    float real() const
    {
        return x;
    }
    float imag() const
    {
        return y;
    }
    void real(float r)
    {
        x = r;
    }
    void imag(float i)
    {
        y = i;
    }
} hipsolverComplex;

typedef struct hipsolverDoubleComplex
{
private:
    double x, y;

public:
    hipsolverDoubleComplex() = default;

    hipsolverDoubleComplex(double r, double i = 0)
        : x(r)
        , y(i)
    {
    }

    double real() const
    {
        return x;
    }
    double imag() const
    {
        return y;
    }
    void real(double r)
    {
        x = r;
    }
    void imag(double i)
    {
        y = i;
    }
} hipsolverDoubleComplex;

inline hipsolverComplex& operator+=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        += reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator+=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        += reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator+(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs += rhs;
}

inline hipsolverDoubleComplex operator+(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs += rhs;
}

inline hipsolverComplex& operator-=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        -= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator-=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        -= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator-(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs -= rhs;
}

inline hipsolverDoubleComplex operator-(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs -= rhs;
}

inline hipsolverComplex& operator*=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        *= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator*=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        *= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator*(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs *= rhs;
}

inline hipsolverDoubleComplex operator*(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs *= rhs;
}

inline hipsolverComplex& operator/=(hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    reinterpret_cast<std::complex<float>&>(lhs)
        /= reinterpret_cast<const std::complex<float>&>(rhs);
    return lhs;
}

inline hipsolverDoubleComplex& operator/=(hipsolverDoubleComplex&       lhs,
                                          const hipsolverDoubleComplex& rhs)
{
    reinterpret_cast<std::complex<double>&>(lhs)
        /= reinterpret_cast<const std::complex<double>&>(rhs);
    return lhs;
}

inline hipsolverComplex operator/(hipsolverComplex lhs, const hipsolverComplex& rhs)
{
    return lhs /= rhs;
}

inline hipsolverDoubleComplex operator/(hipsolverDoubleComplex        lhs,
                                        const hipsolverDoubleComplex& rhs)
{
    return lhs /= rhs;
}

inline bool operator==(const hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    return reinterpret_cast<const std::complex<float>&>(lhs)
           == reinterpret_cast<const std::complex<float>&>(rhs);
}

inline bool operator!=(const hipsolverComplex& lhs, const hipsolverComplex& rhs)
{
    return !(lhs == rhs);
}

inline bool operator==(const hipsolverDoubleComplex& lhs, const hipsolverDoubleComplex& rhs)
{
    return reinterpret_cast<const std::complex<double>&>(lhs)
           == reinterpret_cast<const std::complex<double>&>(rhs);
}

inline bool operator!=(const hipsolverDoubleComplex& lhs, const hipsolverDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

inline hipsolverComplex operator-(const hipsolverComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipsolverDoubleComplex operator-(const hipsolverDoubleComplex& x)
{
    return {-x.real(), -x.imag()};
}

inline hipsolverComplex operator+(const hipsolverComplex& x)
{
    return x;
}

inline hipsolverDoubleComplex operator+(const hipsolverDoubleComplex& x)
{
    return x;
}

namespace std
{
    inline float real(const hipsolverComplex& z)
    {
        return z.real();
    }

    inline double real(const hipsolverDoubleComplex& z)
    {
        return z.real();
    }

    inline float imag(const hipsolverComplex& z)
    {
        return z.imag();
    }

    inline double imag(const hipsolverDoubleComplex& z)
    {
        return z.imag();
    }

    inline hipsolverComplex conj(const hipsolverComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline hipsolverDoubleComplex conj(const hipsolverDoubleComplex& z)
    {
        return {z.real(), -z.imag()};
    }

    inline float abs(const hipsolverComplex& z)
    {
        return abs(reinterpret_cast<const complex<float>&>(z));
    }

    inline double abs(const hipsolverDoubleComplex& z)
    {
        return abs(reinterpret_cast<const complex<double>&>(z));
    }

    inline float conj(const float& r)
    {
        return r;
    }

    inline double conj(const double& r)
    {
        return r;
    }
}
