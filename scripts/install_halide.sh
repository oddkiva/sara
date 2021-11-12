#!/bin/bash
URL=https://github.com/halide/Halide/releases/download/v13.0.0/Halide-13.0.0-x86-64-linux-c3641b6850d156aff6bb01a9c01ef475bd069a31.tar.gz
FILENAME=$(basename -- "${URL}")
FOLDER_NAME=Halide-13.0.0-x86-64-linux

if [[ -d ${HOME}/opt ]]; then
  mkdir ${HOME}/opt
fi
wget ${URL}
tar xvzf ${FILENAME}
mv $FOLDER_NAME ${HOME}/opt/${FOLDER_NAME}

