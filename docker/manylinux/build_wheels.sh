#!/bin/bash

set -e -x

GRANNE_DIR="/granne/"

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
for PYBIN in /opt/python/cp{37,36,35}*/bin; do
    (
        export PATH="$PYBIN:$PATH"
        pip install setuptools setuptools-rust
        pip wheel $GRANNE_DIR -w $GRANNE_DIR/wheels/
    )
done
