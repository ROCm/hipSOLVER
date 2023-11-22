# Changelog for hipSOLVER

Documentation for hipSOLVER is available at
[https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/).

## hipSOLVER 2.0.0 for ROCm 6.0.0

### Additions

* Added hipBLAS as an optional dependency to `hipsolver-test`
  * You can use the `BUILD_HIPBLAS_TESTS` CMake option to test the compatibility between hipSOLVER
    and hipBLAS

### Changes

* The `hipsolverOperation_t` type is now an alias of `hipblasOperation_t`
* The `hipsolverFillMode_t` type is now an alias of `hipblasFillMode_t`
* The `hipsolverSideMode_t` type is now an alias of `hipblasSideMode_t`
* Relax array length requirements for GESVDA

### Fixes

* Tests for hipSOLVER info updates in `ORGBR/UNGBR`, `ORGQR/UNGQR`, `ORGTR/UNGTR`,
  `ORMQR/UNMQR`, and `ORMTR/UNMTR`
* Incorrect singular vectors returned from GESVDA


## hipSOLVER 1.8.1 for ROCm 5.7.0

### Changes

* Changed `hipsolver-test` sparse input data search paths to be relative to the test executable

## hipSOLVER 1.8.0 for ROCm 5.6.0

### Additions

* Added
  [`hipsolverRf*`](https://rocm.docs.amd.com/projects/hipSOLVER/en/docs-5.6.1/userguide/usage.html#porting-cusolver-applications-to-hipsolver)
  APIs for compatibility

## hipSOLVER 1.7.0 for ROCm 5.5.0

### Additions

* Added the following functions:
  * `gesvdj`
    * `hipsolverSgesvdj`, `hipsolverDgesvdj`, `hipsolverCgesvdj`, and `hipsolverZgesvdj`
    * `hipsolverSgesvdj_bufferSize`, `hipsolverDgesvdj_bufferSize`, `hipsolverCgesvdj_bufferSize`, and
      `hipsolverZgesvdj_bufferSize`
  * `gesvdjBatched`
    * `hipsolverSgesvdjBatched`, `hipsolverDgesvdjBatched`, `hipsolverCgesvdjBatched`, and `hipsolverZgesvdjBatched`
    * `hipsolverSgesvdjBatched_bufferSize`, `hipsolverDgesvdjBatched_bufferSize`, `hipsolverCgesvdjBatched_bufferSize`, and `hipsolverZgesvdjBatched_bufferSize`

## hipSOLVER 1.6.0 for ROCm 5.4.0

### Additions

* Added the following compatibility-only functions:
  * `gesvdaStridedBatched`
    * `hipsolverDnSgesvdaStridedBatched`, `hipsolverDnDgesvdaStridedBatched`,
      `hipsolverDnCgesvdaStridedBatched`, and `hipsolverDnZgesvdaStridedBatched`
    * `hipsolverDnSgesvdaStridedBatched_bufferSize`, `hipsolverDnDgesvdaStridedBatched_bufferSize`,
      `hipsolverDnCgesvdaStridedBatched_bufferSize`, and
      `hipsolverDnZgesvdaStridedBatched_bufferSize`

## hipSOLVER 1.5.0 for ROCm 5.3.0

### Additions

* Added the following functions:
  * `syevj`
    * `hipsolverSsyevj`, `hipsolverDsyevj`, `hipsolverCheevj`, and `hipsolverZheevj`
    * `hipsolverSsyevj_bufferSize`, `hipsolverDsyevj_bufferSize`, `hipsolverCheevj_bufferSize`, and
      `hipsolverZheevj_bufferSize`
  * `syevjBatched`
    * `hipsolverSsyevjBatched`, `hipsolverDsyevjBatched`, `hipsolverCheevjBatched`, and
      `hipsolverZheevjBatched`
    * `hipsolverSsyevjBatched_bufferSize`, `hipsolverDsyevjBatched_bufferSize`,
      `hipsolverCheevjBatched_bufferSize`, and `hipsolverZheevjBatched_bufferSize`
  * `sygvj`
    * `hipsolverSsygvj`, `hipsolverDsygvj`, `hipsolverChegvj`, and `hipsolverZhegvj`
    * `hipsolverSsygvj_bufferSize`, `hipsolverDsygvj_bufferSize`, `hipsolverChegvj_bufferSize`, and
      `hipsolverZhegvj_bufferSize`
* Added the following compatibility-only functions:
  * `syevdx/heevdx`
    * `hipsolverDnSsyevdx`, `hipsolverDnDsyevdx`, `hipsolverDnCheevdx`, and `hipsolverDnZheevdx`
    * `hipsolverDnSsyevdx_bufferSize`, `hipsolverDnDsyevdx_bufferSize`,
      `hipsolverDnCheevdx_bufferSize`, and `hipsolverDnZheevdx_bufferSize`
  * `sygvdx/hegvdx`
    * `hipsolverDnSsygvdx`, `hipsolverDnDsygvdx`, `hipsolverDnChegvdx`, and `hipsolverDnZhegvdx`
    * `hipsolverDnSsygvdx_bufferSize`, `hipsolverDnDsygvdx_bufferSize`,
      `hipsolverDnChegvdx_bufferSize`, and `hipsolverDnZhegvdx_bufferSize`
* Added the `--mem_query` option to `hipsolver-bench`, which prints the amount of device memory
  workspace required by the function

### Changes

* The rocSOLVER backend now sets `info` to zero if rocSOLVER doesn't reference `info` (this applies to:
  `orgbr`/`ungbr`, `orgqr`/`ungqr`, `orgtr`/`ungtr`, `ormqr`/`unmqr`, `ormtr`/`unmtr`, `gebrd`, `geqrf`,
  `getrs`, `potrs`, and `sytrd`/`hetrd`)
* `gesvdj` no longer requires extra workspace to transpose `V` when `jobz` is
  `HIPSOLVER_EIG_MODE_VECTOR` and `econ` is 1

### Fixes

* Fixed Fortran return value declarations within `hipsolver_module.f90`
* Fixed `gesvdj_bufferSize` returning `HIPSOLVER_STATUS_INVALID_VALUE` when `jobz` is
  `HIPSOLVER_EIG_MODE_NOVECTOR` and 1 <= `ldv` < `n`
* Fixed `gesvdj` returning `HIPSOLVER_STATUS_INVALID_VALUE` when `jobz` is
  `HIPSOLVER_EIG_MODE_VECTOR`, `econ` is 1, and `m` < `n`

## hipSOLVER 1.4.0 for ROCm 5.2.0

### Additions

* Package generation for test and benchmark executables on all supported operating systems using
  CPack
* File/folder reorganization with backward compatibility support using ROCm-CMake wrapper functions

### Fixes

* Fixed the *ReadTheDocs* documentation generation.

## hipSOLVER 1.3.0 for ROCm 5.1.0

### Additions

* Added the following functions:
  * `gels`
    * `hipsolverSSgels`, `hipsolverDDgels`, `hipsolverCCgels`, and `hipsolverZZgels`
    * `hipsolverSSgels_bufferSize`, `hipsolverDDgels_bufferSize`, `hipsolverCCgels_bufferSize`, and
      `hipsolverZZgels_bufferSize`
* Added library version and device information to `hipsolver-test` output
* Added compatibility APIs with the `hipsolverDn` prefix
* Added compatibility-only functions
  * `gesvdj`
    * `hipsolverDnSgesvdj`, `hipsolverDnDgesvdj`, `hipsolverDnCgesvdj`, and `hipsolverDnZgesvdj`
    * `hipsolverDnSgesvdj_bufferSize`, `hipsolverDnDgesvdj_bufferSize`, `hipsolverDnCgesvdj_bufferSize`,
      and `hipsolverDnZgesvdj_bufferSize`
  * `gesvdjBatched`
    * `hipsolverDnSgesvdjBatched`, `hipsolverDnDgesvdjBatched`, `hipsolverDnCgesvdjBatched`, and
      `hipsolverDnZgesvdjBatched`
    * `hipsolverDnSgesvdjBatched_bufferSize`,` hipsolverDnDgesvdjBatched_bufferSize`,
      `hipsolverDnCgesvdjBatched_bufferSize`, and `hipsolverDnZgesvdjBatched_bufferSize`
  * `syevj`
    * `hipsolverDnSsyevj`, `hipsolverDnDsyevj`, `hipsolverDnCheevj`, and `hipsolverDnZheevj`
    * `hipsolverDnSsyevj_bufferSize`, `hipsolverDnDsyevj_bufferSize`, `hipsolverDnCheevj_bufferSize`, and
      `hipsolverDnZheevj_bufferSize`
  * `syevjBatched`
    * `hipsolverDnSsyevjBatched`, `hipsolverDnDsyevjBatched`, `hipsolverDnCheevjBatched`, and
      `hipsolverDnZheevjBatched`
    * `hipsolverDnSsyevjBatched_bufferSize`, `hipsolverDnDsyevjBatched_bufferSize`,
      `hipsolverDnCheevjBatched_bufferSize`, and `hipsolverDnZheevjBatched_bufferSize`
  * `sygvj`
    * `hipsolverDnSsygv`j, `hipsolverDnDsygvj`, `hipsolverDnChegvj`, and `hipsolverDnZhegvj`
    * `hipsolverDnSsygvj_bufferSize`, `hipsolverDnDsygvj_bufferSize`, `hipsolverDnChegvj_bufferSize`, and
      `hipsolverDnZhegvj_bufferSize`

### Changes

* The rocSOLVER backend now allows `hipsolverXXgels` and `hipsolverXXgesv` to be called in-place
  when B == X
* The rocSOLVER backend now allows `rwork` to be passed as a null pointer to `hipsolverXgesvd`

### Fixes

* `bufferSize` functions will now return `HIPSOLVER_STATUS_NOT_INITIALIZED` instead of
  `HIPSOLVER_STATUS_INVALID_VALUE` when both handle and lwork are null
* Fixed rare memory allocation failure in `syevd`/`heevd` and `sygvd`/`hegvd` caused by improper
  workspace array allocation outside of rocSOLVER.

## hipSOLVER 1.2.0 for ROCm 5.0.0

### Additions

* Added functions
  * `sytrf`
    * `hipsolverSsytrf`, `hipsolverDsytrf`, `hipsolverCsytrf`, and `hipsolverZsytrf`
    * `hipsolverSsytrf_bufferSize`, `hipsolverDsytrf_bufferSize`, `hipsolverCsytrf_bufferSize`, and
      `hipsolverZsytrf_bufferSize`

### Fixes

* Fixed use of incorrect `HIP_PATH` when building from source
  ([GitHub issue #40](https://github.com/ROCmSoftwarePlatform/hipSOLVER/issues/40)).

## hipSOLVER 1.1.0 for ROCm 4.5.0

### Additions

* Added the following functions:
  * `gesv`
    * `hipsolverSSgesv`, `hipsolverDDgesv`, `hipsolverCCgesv`, and `hipsolverZZgesv`
    * `hipsolverSSgesv_bufferSize`, `hipsolverDDgesv_bufferSize`, `hipsolverCCgesv_bufferSize`, and
      `hipsolverZZgesv_bufferSize`
  * `potrs`
    * `hipsolverSpotrs`, `hipsolverDpotrs`, `hipsolverCpotrs`, and `hipsolverZpotrs`
    * `hipsolverSpotrs_bufferSize`, `hipsolverDpotrs_bufferSize`, `hipsolverCpotrs_bufferSize`, and
      `hipsolverZpotrs_bufferSize`
  * `potrsBatched`
    * `hipsolverSpotrsBatched`, `hipsolverDpotrsBatched`, `hipsolverCpotrsBatched`, and
      `hipsolverZpotrsBatched`
    * `hipsolverSpotrsBatched_bufferSize`, `hipsolverDpotrsBatched_bufferSize`,
      `hipsolverCpotrsBatched_bufferSize`, and `hipsolverZpotrsBatched_bufferSize`
  * `potri`
    * `hipsolverSpotri`, `hipsolverDpotri`, `hipsolverCpotri`, and `hipsolverZpotri`
    * `hipsolverSpotri_bufferSize`, `hipsolverDpotri_bufferSize`, `hipsolverCpotri_bufferSize`, and
      `hipsolverZpotri_bufferSize`
  * `orgbr/ungbr`
    * `hipsolverSorgbr`, `hipsolverDorgbr`, `hipsolverCungbr`, `hipsolverZungbr`
    * `hipsolverSorgbr_bufferSize`, `hipsolverDorgbr_bufferSize`, `hipsolverCungbr_bufferSize`, and
      `hipsolverZungbr_bufferSize`
  * `orgqr/ungqr`
    * `hipsolverSorgqr`, `hipsolverDorgqr`, `hipsolverCungqr`, and `hipsolverZungqr`
    * `hipsolverSorgqr_bufferSize`, `hipsolverDorgqr_bufferSize`, `hipsolverCungqr_bufferSize`, and
      `hipsolverZungqr_bufferSize`
  * `orgtr/ungtr`
    * `hipsolverSorgqr`, `hipsolverDorgqr`, `hipsolverCungqr`, and `hipsolverZungqr`
    * `hipsolverSorgtr_bufferSize`, `hipsolverDorgtr_bufferSize`, `hipsolverCungtr_bufferSize`, and
      `hipsolverZungtr_bufferSize`
  * `ormqr/unmqr`
    * `hipsolverSormqr`, `hipsolverDormqr`, `hipsolverCunmqr`, and `hipsolverZunmqr`
    * `hipsolverSormqr_bufferSize`, `hipsolverDormqr_bufferSize`, `hipsolverCunmqr_bufferSize`, and
      `hipsolverZunmqr_bufferSize`
  * `ormtr/unmtr`
    * `hipsolverSormtr`, `hipsolverDormtr`, `hipsolverCunmtr`, and `hipsolverZunmtr`
    * `hipsolverSormtr_bufferSize`, `hipsolverDormtr_bufferSize`, `hipsolverCunmtr_bufferSize`, and
      `hipsolverZunmtr_bufferSize`
  * `gebrd`
    * `hipsolverSgebrd`, `hipsolverDgebrd`, `hipsolverCgebrd`, and `hipsolverZgebrd`
    * `hipsolverSgebrd_bufferSize`, `hipsolverDgebrd_bufferSize`, `hipsolverCgebrd_bufferSize`, and
      `hipsolverZgebrd_bufferSize`
  * `geqrf`
    * `hipsolverSgeqrf`, `hipsolverDgeqrf`, `hipsolverCgeqrf`, and `hipsolverZgeqrf`
    * `hipsolverSgeqrf_bufferSize`, `hipsolverDgeqrf_bufferSize`, `hipsolverCgeqrf_bufferSize`, and
      `hipsolverZgeqrf_bufferSize`
  * `gesvd`
    * `hipsolverSgesvd`, `hipsolverDgesvd`, `hipsolverCgesvd`, and `hipsolverZgesvd`
    * `hipsolverSgesvd_bufferSize`, `hipsolverDgesvd_bufferSize`, `hipsolverCgesvd_bufferSize`, and
      `hipsolverZgesvd_bufferSize`
  * `getrs`
    * `hipsolverSgetrs`, `hipsolverDgetrs`, `hipsolverCgetrs`, and `hipsolverZgetrs`
    * `hipsolverSgetrs_bufferSize, hipsolverDgetrs_bufferSize, hipsolverCgetrs_bufferSize, hipsolverZgetrs_bufferSize
  * `potrf`
    * `hipsolverSpotrf`, `hipsolverDpotrf`, `hipsolverCpotrf`, and `hipsolverZpotrf`
    * `hipsolverSpotrf_bufferSize`, `hipsolverDpotrf_bufferSize`, `hipsolverCpotrf_bufferSize`, and
      `hipsolverZpotrf_bufferSize`
  * `potrfBatched`
    * `hipsolverSpotrfBatched`, `hipsolverDpotrfBatched`, `hipsolverCpotrfBatched`, and
      `hipsolverZpotrfBatched`
    * `hipsolverSpotrfBatched_bufferSize`, `hipsolverDpotrfBatched_bufferSize`,
      `hipsolverCpotrfBatched_bufferSize`, and `hipsolverZpotrfBatched_bufferSize`
  * `syevd/heevd`
    * `hipsolverSsyevd, hipsolverDsyevd` and `hipsolverCheevd, hipsolverZheevd`
    * `hipsolverSsyevd_bufferSize`, `hipsolverDsyevd_bufferSize`, `hipsolverCheevd_bufferSize`, and
      `hipsolverZheevd_bufferSize`
  * `sygvd/hegvd`
    * `hipsolverSsygvd`, `hipsolverDsygvd`, `hipsolverChegvd`, and `hipsolverZhegvd`
    * `hipsolverSsygvd_bufferSize`, `hipsolverDsygvd_bufferSize`, `hipsolverChegvd_bufferSize`, and
      `hipsolverZhegvd_bufferSize`
  * `sytrd/hetrd`
    * `hipsolverSsytrd`, `hipsolverDsytrd`, `hipsolverChetrd`, and `hipsolverZhetrd`
    * `hipsolverSsytrd_bufferSize`, `hipsolverDsytrd_bufferSize`, `hipsolverChetrd_bufferSize`, and
      `hipsolverZhetrd_bufferSize`
  * `getrf`
    * `hipsolverSgetrf`, `hipsolverDgetrf`, `hipsolverCgetrf`, and `hipsolverZgetrf`
    * `hipsolverSgetrf_bufferSize`, `hipsolverDgetrf_bufferSize`, `hipsolverCgetrf_bufferSize`, and
      `hipsolverZgetrf_bufferSize`
  * `auxiliary`
    * `hipsolverCreate` and `hipsolverDestroy`
    * `hipsolverSetStream` and `hipsolverGetStream`

### Changes

* hipSOLVER functions now return `HIPSOLVER_STATUS_INVALID_ENUM` or
  `HIPSOLVER_STATUS_UNKNOWN` status codes rather than throw exceptions
* `hipsolverXgetrf` functions now take `lwork` as an argument

### Removals

* Removed unused `HIPSOLVER_FILL_MODE_FULL` enum value.
* Removed `hipsolverComplex` and `hipsolverDoubleComplex` from the library; use `hipFloatComplex`
  and `hipDoubleComplex` instead
