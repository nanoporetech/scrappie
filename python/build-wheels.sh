#!/bin/bash
set -e -x
export MANYLINUX=1

# Install a system package required by our library
yum install -y atlas-devel
ln -s /usr/lib64/atlas/libcblas.so /usr/lib64/libblas.so

cd /io/python

# Compile wheels
for minor in 4 5 6; do
    PYBIN="/opt/python/cp3${minor}-cp3${minor}m/bin"
    "${PYBIN}/pip" wheel . -w wheelhouse/
done


# Bundle external shared libraries into the wheels
for whl in wheelhouse/scrappy*.whl; do
    auditwheel repair "$whl" -w wheelhouse/
done


# Install packages and "test"
for minor in 4 5 6; do
    PYBIN="/opt/python/cp3${minor}-cp3${minor}m/bin"
    "${PYBIN}/pip" install scrappy --no-index -f wheelhouse
    "${PYBIN}/python" -c "from scrappy import *; import numpy as np; print(basecall_raw(np.random.normal(10,4,1000)))" 
done
