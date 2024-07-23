/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsparse.hpp"
#include "load_function.hpp"

HIPSOLVER_BEGIN_NAMESPACE

fp_rocsparse_create_mat_descr   g_rocsparse_create_mat_descr;
fp_rocsparse_destroy_mat_descr  g_rocsparse_destroy_mat_descr;
fp_rocsparse_get_mat_type       g_rocsparse_get_mat_type;
fp_rocsparse_get_mat_index_base g_rocsparse_get_mat_index_base;

static bool load_rocsparse()
{
#ifdef _WIN32
    // Library users will need to call SetErrorMode(SEM_FAILCRITICALERRORS) if
    // they wish to avoid an error message box when this library is not found.
    // The call is not done by hipSOLVER directly, as it is not thread-safe and
    // will affect the global state of the program.
    void* handle = LoadLibraryW(L"librocsparse.dll");
#else
    void* handle = dlopen("librocsparse.so.1", RTLD_NOW | RTLD_LOCAL);
    char* err    = dlerror(); // clear errors
#ifndef NDEBUG
    if(!handle)
        std::cerr << "hipsolver: error loading librocsparse.so.1: " << err << std::endl;
#endif
#endif /* _WIN32 */
    if(!handle)
        return false;

    if(!load_function(handle, "rocsparse_create_mat_descr", g_rocsparse_create_mat_descr))
        return false;
    if(!load_function(handle, "rocsparse_destroy_mat_descr", g_rocsparse_destroy_mat_descr))
        return false;
    if(!load_function(handle, "rocsparse_get_mat_type", g_rocsparse_get_mat_type))
        return false;
    if(!load_function(handle, "rocsparse_get_mat_index_base", g_rocsparse_get_mat_index_base))
        return false;

    return true;
}

bool try_load_rocsparse()
{
    // Function-scope static initialization has been thread-safe since C++11.
    // There is an implicit mutex guarding the initialization.
    static bool result = load_rocsparse();
    return result;
}

HIPSOLVER_END_NAMESPACE
