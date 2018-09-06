FROM rust:latest AS build

COPY requirements-linux.txt /tmp/

RUN set -ex \
    && apt-get update \
    && cat /tmp/requirements-linux.txt | awk -F: '{print $1}' | xargs apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm /tmp/requirements-linux.txt

RUN update-alternatives --config libblas.so.3

WORKDIR /opt/granne
ENV RUSTFLAGS=-L/usr/lib/openblas-base/
ENV CARGO_BLAS_TYPE=static
COPY . ./
RUN cargo build --release

FROM debian:stretch-slim
COPY --from=build /opt/granne/target/release/build_index /usr/bin/granne
COPY --from=build /opt/granne/target/release/generate_queries /usr/bin/generate_queries
COPY --from=build /opt/granne/target/release/generate_query_vectors /usr/bin/generate_query_vectors
VOLUME ["/data"]
ENTRYPOINT ["granne"]
CMD ["--help"]
