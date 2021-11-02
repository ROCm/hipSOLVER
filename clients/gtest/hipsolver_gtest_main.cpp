/* ************************************************************************
 * Copyright 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdlib>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <hipsolver.h>

#include "utility.hpp"

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

static void print_version_info()
{
    // clang-format off
    std::cout << "hipSOLVER version "
        STRINGIFY(hipsolverVersionMajor) "."
        STRINGIFY(hipsolverVersionMinor) "."
        STRINGIFY(hipsolverVersionPatch) "."
        STRINGIFY(hipsolverVersionTweak)
        << std::endl;
    // clang-format on
}

/* =====================================================================
      Main function:
=================================================================== */

int main(int argc, char** argv)
{
    print_version_info();

    // print device info
    int device_count = query_device_property();
    if(device_count <= 0)
    {
        std::cerr << "Error: No devices found" << std::endl;
        return EXIT_FAILURE;
    }
    set_device(0); // use first device

    ::testing::InitGoogleTest(&argc, argv);

    int status = RUN_ALL_TESTS();
    print_version_info(); // redundant, but convenient when tests fail
    return status;
}
