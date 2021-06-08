#!/bin/bash

# This script needs to be edited to bump new master version to new develop for new release.
# - run this script after running bump_master_version.sh and merging develop into master
# - run this script in master branch
# - after running this script merge master into develop

OLD_HIPSOLVER_VERSION="1.0.0"
NEW_HIPSOLVER_VERSION="1.1.0"

OLD_HIPSOLVER_DOCS_VERSION="1.0"
NEW_HIPSOLVER_DOCS_VERSION="1.1"

OLD_MINIMUM_ROCBLAS_VERSION="2.40.0"
NEW_MINIMUM_ROCBLAS_VERSION="2.41.0"

OLD_MINIMUM_ROCSOLVER_VERSION="3.14.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.15.0"

sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt
sed -i "s/${OLD_HIPSOLVER_DOCS_VERSION}/${NEW_HIPSOLVER_DOCS_VERSION}/g" docs/source/conf.py
sed -i "s/${OLD_MINIMUM_ROCBLAS_VERSION}/${NEW_MINIMUM_ROCBLAS_VERSION}/g" library/CMakeLists.txt
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" library/CMakeLists.txt
