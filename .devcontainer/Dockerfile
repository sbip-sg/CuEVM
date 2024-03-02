FROM golang:latest as golang-builder
RUN git clone https://github.com/cassc/goevmlab --depth 1
RUN cd goevmlab && \
  go build ./cmd/generic-fuzzer && \
  go build ./cmd/checkslow && \
  go build ./cmd/minimizer && \
  go build ./cmd/repro && \
  go build ./cmd/runtest && \
  go build ./cmd/tracediff && \
  go build ./cmd/traceview
RUN git clone https://github.com/ethereum/go-ethereum --depth 1
RUN cd go-ethereum && go run build/ci.go install -static ./cmd/evm


FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# LABEL about the custom image
LABEL maintainer="dancioc@nus.edu.sg"
LABEL version="0.1"
LABEL description="This is a custom Docker Image for CuEVM project"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive
# display
ENV DISPLAY=host.docker.internal:0.0
# timezone
ENV TZ=Asia/Singapore
# Update Ubuntu Software repository
# necasry tools curl zip unzip git wget
# gmp libgmp-dev
# cjson libcjson1 libcjson-dev
# clang
# valgrind
RUN apt-get update && apt-get upgrade -y && apt-get install -y locales && locale-gen "en_US.UTF-8" && dpkg-reconfigure locales && apt-get install -y curl zip unzip git wget libgmp-dev libcjson1 libcjson-dev libclang-dev valgrind clang-format
# install rustup for REVMI
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o sh.rustup.rs && sh sh.rustup.rs -y && rm sh.rustup.rs
# install tools for documentation
RUN apt-get update && apt-get install -y doxygen sphinx cmake python3-pip && pip3 install breathe exhale sphinx_rtd_theme sphinx_book_theme

# Go-evmlab targets
COPY --from=golang-builder /go/goevmlab/generic-fuzzer /
COPY --from=golang-builder /go/goevmlab/checkslow  /
COPY --from=golang-builder /go/goevmlab/minimizer /
COPY --from=golang-builder /go/goevmlab/repro /
COPY --from=golang-builder /go/goevmlab/runtest /
COPY --from=golang-builder /go/goevmlab/tracediff /
COPY --from=golang-builder /go/goevmlab/traceview /

COPY --from=golang-builder /go/go-ethereum/build/bin/evm /gethvm