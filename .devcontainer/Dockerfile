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
RUN apt-get update && apt-get upgrade -y && apt-get install -y locales && locale-gen "en_US.UTF-8" && dpkg-reconfigure locales && apt-get install -y curl zip unzip git wget libgmp-dev libcjson1 libcjson-dev libclang-dev valgrind
# install rustup for REVMI
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o sh.rustup.rs && sh sh.rustup.rs -y && rm sh.rustup.rs