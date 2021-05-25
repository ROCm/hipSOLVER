# Change Log for hipSOLVER


## [(Unreleased) hipSOLVER 1.0.0 for ROCm 4.4]
### Added
- Added functions
  - orgqr/ungqr
    - hipsolverSorgqr_bufferSize, hipsolverDorgqr_bufferSize, hipsolverCungqr_bufferSize, hipsolverZungqr_bufferSize
    - hipsolverSorgqr, hipsolverDorgqr, hipsolverCungqr, hipsolverZungqr
  - orgtr/ungtr
    - hipsolverSorgtr_bufferSize, hipsolverDorgtr_bufferSize, hipsolverCungtr_bufferSize, hipsolverZungtr_bufferSize
    - hipsolverSorgtr, hipsolverDorgtr, hipsolverCungtr, hipsolverZungtr
  - ormqr/unmqr
    - hipsolverSormqr_bufferSize, hipsolverDormqr_bufferSize, hipsolverCunmqr_bufferSize, hipsolverZunmqr_bufferSize
    - hipsolverSormqr, hipsolverDormqr, hipsolverCunmqr, hipsolverZunmqr
  - gebrd
    - hipsolverSgebrd_bufferSize, hipsolverDgebrd_bufferSize, hipsolverCgebrd_bufferSize, hipsolverZgebrd_bufferSize
    - hipsolverSgebrd, hipsolverDgebrd, hipsolverCgebrd, hipsolverZgebrd
  - geqrf
    - hipsolverSgeqrf_bufferSize, hipsolverDgeqrf_bufferSize, hipsolverCgeqrf_bufferSize, hipsolverZgeqrf_bufferSize
    - hipsolverSgeqrf, hipsolverDgeqrf, hipsolverCgeqrf, hipsolverZgeqrf
  - getrs
    - hipsolverSgetrs, hipsolverDgetrs, hipsolverCgetrs, hipsolverZgetrs
  - potrf
    - hipsolverSpotrf_bufferSize, hipsolverDpotrf_bufferSize, hipsolverCpotrf_bufferSize, hipsolverZpotrf_bufferSize
    - hipsolverSpotrf, hipsolverDpotrf, hipsolverCpotrf, hipsolverZpotrf
  - potrfBatched
    - hipsolverSpotrfBatched, hipsolverDpotrfBatched, hipsolverCpotrfBatched, hipsolverZpotrfBatched
  - sytrd/hetrd
    - hipsolverSsytrd_bufferSize, hipsolverDsytrd_bufferSize, hipsolverChetrd_bufferSize, hipsolverZhetrd_bufferSize
    - hipsolverSsytrd, hipsolverDsytrd, hipsolverChetrd, hipsolverZhetrd

### Changed
- hipSOLVER functions will now return HIPSOLVER_STATUS_INVALID_ENUM or HIPSOLVER_STATUS_UNKNOWN status codes rather than throw exceptions

### Removed
- Removed unused HIPSOLVER_FILL_MODE_FULL enum value.


## [hipSOLVER 0.1.0 for ROCm 4.2]
### Added
- Created hipSOLVER repository
- Added functions
  - auxiliary
    - hipsolverCreate, hipsolverDestroy
    - hipsolverSetStream, hipsolverGetStream
  - getrf
    - hipsolverSgetrf_bufferSize, hipsolverDgetrf_bufferSize, hipsolverCgetrf_bufferSize, hipsolverZgetrf_bufferSize
    - hipsolverSgetrf, hipsolverDgetrf, hipsolverCgetrf, hipsolverZgetrf
