# Change Log for hipSOLVER

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
