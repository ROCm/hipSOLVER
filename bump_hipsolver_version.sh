#!/bin/bash

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for hipSOLVER version string
OLD_HIPSOLVER_VERSION="2.3.0"
NEW_HIPSOLVER_VERSION="2.4.0"
sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt

# for rocBLAS/rocSOLVER package requirements
OLD_MINIMUM_ROCBLAS_VERSION="4.3.0"
NEW_MINIMUM_ROCBLAS_VERSION="4.4.0"
OLD_MINIMUM_ROCSOLVER_VERSION="3.27.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.28.0"
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" CMakeLists.txt
