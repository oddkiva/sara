FROM nvidia/cudagl:11.4.2-devel

MAINTAINER "David OK" <david.ok8@gmail.com>

# To avoid console interaction with apt.
ARG DEBIAN_FRONTEND=noninteractive

ADD . /opt/sara
WORKDIR /opt/sara

RUN bash ./scripts/install_ubuntu_dependencies.sh

RUN pip3 install -r requirements.txt
RUN bash ./scripts/install_halide.sh
# RUN bash ./build.sh Debug
