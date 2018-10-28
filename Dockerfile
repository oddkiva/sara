FROM ubuntu:14.04


# Install pre-requisites.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      lcov \
      libboost-test-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      libqt5opengl5 \
      libqt5opengl5-dev \
      qtbase5-dev \
      wget \
      yasm && \
    rm -rf /var/lib/apt/lists/*

# Build and install third-party software.
COPY ./scripts/travis-build-and-install-ffmpeg.sh \
     /root/travis-build-and-install-ffmpeg.sh

WORKDIR /root
RUN /root/travis-build-and-install-ffmpeg.sh


# Simulate travis build.
COPY ./scripts/travis-build.sh \
     /root/travis-build.sh
RUN git clone https://github.com/DO-CV/sara
