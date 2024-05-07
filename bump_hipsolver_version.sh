#!/bin/bash

# run this script in develop after creating release-staging branch for feature-complete date
# Edit script to bump versions for new development cycle/release.

# for hipSOLVER version string
OLD_HIPSOLVER_VERSION="2.3.0"
NEW_HIPSOLVER_VERSION="2.4.0"
sed -i "s/${OLD_HIPSOLVER_VERSION}/${NEW_HIPSOLVER_VERSION}/g" CMakeLists.txt

OLD_HIPSOLVER_SOVERSION="0.2"
NEW_HIPSOLVER_SOVERSION="0.3"
sed -i "s/${OLD_HIPSOLVER_SOVERSION}/${NEW_HIPSOLVER_SOVERSION}/g" library/CMakeLists.txt

# for rocSOLVER package requirements
OLD_MINIMUM_ROCSOLVER_VERSION="3.27.0"
NEW_MINIMUM_ROCSOLVER_VERSION="3.28.0"
sed -i "s/${OLD_MINIMUM_ROCSOLVER_VERSION}/${NEW_MINIMUM_ROCSOLVER_VERSION}/g" CMakeLists.txt
