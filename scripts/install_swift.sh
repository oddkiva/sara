#!/bin/bash
URL=https://download.swift.org/swift-5.5.1-release/ubuntu2004/swift-5.5.1-RELEASE/swift-5.5.1-RELEASE-ubuntu20.04.tar.gz
FILENAME=$(basename -- "${URL}")

if [[ -d ${HOME}/opt ]]; then
  mkdir ${HOME}/opt
fi
wget https://download.swift.org/swift-5.5.1-release/ubuntu2004/swift-5.5.1-RELEASE/swift-5.5.1-RELEASE-ubuntu20.04.tar.gz
tar xvzf ${FILENAME}
mv ${FILENAME} ${HOME}/opt
