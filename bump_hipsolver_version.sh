#!/bin/bash

# run this script in develop after merging develop/staging into master at the feature-complete date
# Edit script to bump versions for new development cycle/release.

OLD_HIPSOLVER_VERSION="1.2.0"  # Distributed with ROCm 5.0
NEW_HIPSOLVER_VERSION="1.3.0"  # Distributed with ROCm 5.1
sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt

# for documentation
OLD_HIPSOLVER_DOCS_VERSION="1.2"
NEW_HIPSOLVER_DOCS_VERSION="1.3"
sed -i "s/${OLD_HIPSOLVER_DOCS_VERSION}/${NEW_HIPSOLVER_DOCS_VERSION}/g" docs/source/conf.py

# for rocBLAS/rocSOLVER package requirements
OLD_MINIMUM_ROCBLAS_VERSION="2.42.0"
NEW_MINIMUM_ROCBLAS_VERSION="2.43.0"
OLD_MINIMUM_ROCSOLVER_VERSION="3.16.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.17.0"
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" library/CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" library/CMakeLists.txt
