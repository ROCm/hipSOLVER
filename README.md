# hipSOLVER

hipSOLVER is a LAPACK marshalling library with multiple supported backends. It sits between your
application and a 'worker' LAPACK library, where it marshals inputs to the backend library and marshals
results to your application. hipSOLVER exports an interface that doesn't require the client to change,
regardless of the chosen backend.

hipSOLVER supports [rocSOLVER](https://github.com/ROCmSoftwarePlatform/rocSOLVER) and
[cuSOLVER](https://developer.nvidia.com/cusolver) backends.

## Documentation

Documentation for hipSOLVER is available at
[https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/).

To build documentation locally, use the following code:

```shell
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Build and install

To download the hipSOLVER source code, clone this repository using:

```bash
git clone https://github.com/ROCmSoftwarePlatform/hipSOLVER.git
```

To install hipSOLVER, you must have must install these dependencies:

* rocSOLVER
  *rocBLAS
    * Tensile
  * rocSPARSE
    * rocPRIM
* SuiteSparse

Once these are installed, use the following commands to build and install hipSOLVER:

```bash
cd hipSOLVER
./install.sh -i
```

The install directory is `/opt/rocm`.

Once installed, hipSOLVER can be used just like any other C API library. Include the header file in your
code to make the hipSOLVER library a link-time and run-time dependency for your application.

## Using the hipSOLVER interface

The hipSOLVER interface is compatible with rocSOLVER and cuSOLVER-v11 APIs. Porting a CUDA
application that originally calls the cuSOLVER API to an application that calls the hipSOLVER API is
relatively straightforward; refer to
[porting a cuSOLVER application to hipSOLVER](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/userguide/usage.html#porting-cusolver-applications-to-hipsolver).

For example, the hipSOLVER SGEQRF interface is

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

## Supported functionality

For a complete list supported functions, refer to the
[hipSOLVER user guide](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/userguide/index.html)
and [API documentation](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/api/index.html).
