FROM centos:7.0.1406
MAINTAINER Tim Massingham <tim.massingham@nanoporetech.com>
RUN yum install -y epel-release
RUN yum install -y gcc git make cmake

RUN yum install -y gcc CUnit CUnit-devel hdf5 hdf5-devel openblas openblas-devel
# For cblas.h
RUN yum install -y atlas-devel
RUN ln -s /usr/lib64/libopenblaso.so /usr/lib64/libblas.so

RUN git clone --depth 1 http://github.com/nanoporetech/scrappie.git

RUN cd scrappie && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make test

