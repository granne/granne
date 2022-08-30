#!/usr/bin/env bash
set -euxo pipefail

echo "PWD ${PWD}"

rm -rf dist || true

# Cross-platform way to create a temporary dir.
mytmpdir=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')

# Build x86 wheels, put in a temp folder.
docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_x86_64 /io/build-wheels.sh
mv dist "${mytmpdir}/dist_x86_64"

# Build ARM wheels, put in a temp folder.
docker run --rm -v `pwd`:/io quay.io/pypa/manylinux2014_aarch64 /io/build-wheels.sh
mv dist "${mytmpdir}/dist_aarch64"

# Move all wheels to dist/
mkdir dist
mv ${mytmpdir}/dist_x86_64/* dist/
mv ${mytmpdir}/dist_aarch64/* dist/
rm -rf "${mytmpdir}/dist_x86_64" "${mytmpdir}/dist_aarch64"