# hipSOLVER
hipSOLVER is a LAPACK marshalling library, with multiple supported backends.  It sits between the application and a 'worker' LAPACK library, marshalling inputs into the backend library and marshalling results back to the application.  hipSOLVER exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipSOLVER supports [rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER) and [cuSOLVER](https://developer.nvidia.com/cusolver) as backends.

## Documentation

For a detailed description of the hipSOLVER library, its implemented routines, the installation process and user guide, see the [hipSOLVER documentation](https://hipsolver.readthedocs.io/en/latest/).

## Quickstart Build

To download the hipSOLVER source code, clone this repository with the command:

    git clone https://github.com/ROCmSoftwarePlatform/hipSOLVER.git

hipSOLVER requires either cuSOLVER or rocSOLVER + rocBLAS to be installed on the system. Once these are installed, the following commands will build hipSOLVER and install to `/opt/rocm`:

    cd hipSOLVER
    ./install.sh -i

Once installed, hipSOLVER can be used just like any other library with a C API. The header file will need to be included in the user code, and the hipSOLVER library will become a link-time and run-time dependency for the user application.

For more information on building and installing hipSOLVER, see the [hipSOLVER install guide](https://hipsolver.readthedocs.io/en/latest/userguide_install.html)

## Using the hipSOLVER Interface
The hipSOLVER interface is compatible with the rocSOLVER and cuSOLVER-v11 APIs. Porting a CUDA application that originally calls the cuSOLVER API to an application calling the hipSOLVER API should be fairly straightforward (see [porting a cuSOLVER application to hipSOLVER](https://hipsolver.readthedocs.io/en/latest/userguide_intro.html#porting-a-cusolver-application-to-hipsolver)). For example, the hipSOLVER SGEQRF interface is

```c
hipsolverStatus_t
hipsolverSgeqrf_bufferSize(hipsolverHandle_t handle,
                           int m,
                           int n,
                           float* A,
                           int lda,
                           int* lwork);
```

```c
hipsolverStatus_t
hipsolverSgeqrf(hipsolverHandle_t handle,
                int               m,
                int               n,
                float*            A,
                int               lda,
                float*            tau,
                float*            work,
                int               lwork,
                int*              devInfo);
```

## Supported Functionality
For a complete listing of all supported functions, see the [hipSOLVER user guide](https://hipsolver.readthedocs.io/en/latest/userguide_intro.html) and/or [API documentation](https://hipsolver.readthedocs.io/en/latest/api_index.html).
