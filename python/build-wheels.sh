#!/bin/bash
set -e -x

BUILD_PREFIX="/usr/local"
OPENBLAS_VERS="0.2.18"
OPENBLAS_TAR="openblas_${OPENBLAS_VERS}.tgz"
OPENBLAS_PATH="/usr/local/lib/libopenblasp-r${OPENBLAS_VERS}.so"
PACKAGE_NAME='scrappie'
# this is picked up by build.py
export SCRAPPIESRC=$(pwd)/src  # pip wheel changes cwd
export MANYLINUX=1

function build_openblas {
    # this takes a long time so record success in openblas-built
    if [ -e "openblas-built" ]; then return; fi
    if [ -d "OpenBLAS" ]; then
        (cd OpenBLAS && git clean -fxd && git reset --hard)
    else
        git clone https://github.com/xianyi/OpenBLAS
    fi
    (cd OpenBLAS \
        && git checkout "v${OPENBLAS_VERSION}" \
        && make DYNAMIC_ARCH=1 USE_OPENMP=0 NUM_THREADS=64 TARGET=NEHALEM > /dev/null)
    touch openblas-built
}

function install_openblas {
    (cd OpenBLAS && make PREFIX=$BUILD_PREFIX install)
    # copy license to a sensible place
    license_path="/usr/share/doc/openblas"
    mkdir -p ${license_path}
    cp OpenBLAS/LICENSE ${license_path}
    tar zcf ${OPENBLAS_TAR} /usr/local/lib /usr/local/include ${license_path}
}

# OpenBLAS possibilities:
#   i) check /usr/local/lib for openblas -> nothing to do
#  ii) look for a tar (containing compiled blas) -> unpack
# iii) build from scratch
if [ -e ${OPENBLAS_PATH} ]; then
    echo "Found OpenBLAS at ${OPENBLAS_PATH}"
elif [ -e ${OPENBLAS_TAR} ]; then
    echo "Unpacking OpenBLAS tar ${OPENBLAS_TAR}"
    tar xzf ${OPENBLAS_TAR} -C /
else
    echo "Building OpenBLAS"
    build_openblas
    install_openblas
fi

# don't do 33 since numpy doesn't like it
PYTHONS=$(ls -1 /opt/python/ | grep -v 33 | xargs -I {} echo /opt/python/{}/bin)

cd python

# Compile wheels
for PYBIN in ${PYTHONS}; do
    echo $PYBIN
    "${PYBIN}/pip" wheel . -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in "wheelhouse/${PACKAGE_NAME}"*.whl; do
    auditwheel repair "${whl}" -w wheelhouse/
    rm "${whl}"
done

# Install packages and "test"
for PYBIN in ${PYTHONS}; do
    "${PYBIN}/pip" install "${PACKAGE_NAME}" --no-index -f wheelhouse
    "${PYBIN}/python" -c "from scrappy import *; import numpy as np; print(basecall_raw(np.random.normal(10,4,1000)))" 
done


