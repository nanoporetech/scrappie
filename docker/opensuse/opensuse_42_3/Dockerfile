FROM opensuse:42.3
MAINTAINER Tim Massingham <tim.massingham@nanoporetech.com>

RUN zypper refresh
RUN zypper --non-interactive install gcc git make cmake
RUN zypper --non-interactive install \
    libopenblas_serial0 libopenblas_serial-devel cblas-devel \
    libcunit1 cunit-devel hdf5 hdf5-devel
RUN ln -s /usr/lib64/libblas.so.3 /usr/lib64/libblas.so

RUN git clone --depth 1 http://github.com/nanoporetech/scrappie.git

RUN cd scrappie && \
    mkdir build && \
    cd build && \ 
    cmake .. && \
    make && \
    make test
