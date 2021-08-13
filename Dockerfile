FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

MAINTAINER "David OK" <david.ok8@gmail.com>

# To avoid console interaction with apt.
ARG DEBIAN_FRONTEND=noninteractive

RUN mkdir /provision
WORKDIR /provision

COPY ./scripts/install_ubuntu_dependencies.sh /provision/install_ubuntu_dependencies.sh
RUN bash /provision/install_ubuntu_dependencies.sh
