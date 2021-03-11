#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_HIPSOLVER_VERSION="0.2.0"
NEW_HIPSOLVER_VERSION="0.3.0"

OLD_MINIMUM_ROCBLAS_VERSION="2.39.0"
NEW_MINIMUM_ROCBLAS_VERSION="2.40.0"

OLD_MINIMUM_ROCSOLVER_VERSION="3.13.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.14.0"

sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" library/CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" library/CMakeLists.txt
