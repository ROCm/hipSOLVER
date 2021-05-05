# Change Log for hipSOLVER


## [(Unreleased) hipSOLVER 1.0.0 for ROCm 4.4]
### Added
- Added functions
  - getrs
    - hipsolverSgetrs, hipsolverDgetrs, hipsolverCgetrs, hipsolverZgetrs
  - potrf
    - hipsolverSpotrf_bufferSize, hipsolverDpotrf_bufferSize, hipsolverCpotrf_bufferSize, hipsolverZpotrf_bufferSize
    - hipsolverSpotrf, hipsolverDpotrf, hipsolverCpotrf, hipsolverZpotrf

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
