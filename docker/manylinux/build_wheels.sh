#!/bin/bash

set -e -x

GRANNE_DIR="/granne/"

for PYBIN in /opt/python/cp{27,35,36}*/bin; do
    (
        export PATH="$PYBIN:$PATH"
        pip install setuptools setuptools-rust
        pip wheel $GRANNE_DIR -w $GRANNE_DIR/wheels/
    )
done
