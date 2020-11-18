#!/bin/bash

# This script needs to be edited to bump old develop version to new master version for new release.
# - run this script in develop branch
# - after running this script merge develop into master
# - after running this script and merging develop into master, run bump_develop_version.sh in master and
#   merge master into develop

OLD_HIPSOLVER_VERSION="0.0.0"
NEW_HIPSOLVER_VERSION="0.1.0"

OLD_MINIMUM_ROCBLAS_VERSION="2.34.0"
NEW_MINIMUM_ROCBLAS_VERSION="2.35.0"

OLD_MINIMUM_ROCSOLVER_VERSION="3.10.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.11.0"

sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" library/CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" library/CMakeLists.txt
