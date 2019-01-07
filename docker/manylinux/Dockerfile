FROM quay.io/pypa/manylinux1_x86_64

RUN yum -y update; yum clean all

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly
ENV PATH="/root/.cargo/bin:$PATH"

RUN yum -y install \
    blas-devel.x86_64 \
    && yum clean all

ENV LD_LIBRARY_PATH=/usr/lib64/

COPY build_wheels.sh /opt/
