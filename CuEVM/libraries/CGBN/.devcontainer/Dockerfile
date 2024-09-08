FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# LABEL about the custom image
LABEL maintainer="stefan_dan.ciocirlan@upb.ro"
LABEL version="0.1"
LABEL description="This is a custom Docker Image for CGBN project"

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive
# display
ENV DISPLAY=host.docker.internal:0.0
# timezone
ENV TZ=Asia/Singapore
# Update Ubuntu Software repository
# necasry tools curl zip unzip git wget
# gmp libgmp-dev
# clang
# valgrind
RUN apt-get update && apt-get upgrade -y && apt-get install -y locales && locale-gen "en_US.UTF-8" && dpkg-reconfigure locales && apt-get install -y curl zip unzip git wget libgmp-dev libclang-dev valgrind clang-format
# install tools for documentation
RUN apt-get update && apt-get install -y doxygen sphinx cmake python3-pip && pip3 install breathe exhale sphinx_rtd_theme sphinx_book_theme
# install tools for testing
RUN apt-get update && apt-get install -y libgtest-dev
# Build GTest library
RUN cd /usr/src/googletest && \
    cmake . && \
    cmake --build . --target install
