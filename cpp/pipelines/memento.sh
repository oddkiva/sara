#!/bin/bash
set -ex

SARA_DIR=$HOME/GitHub/DO-CV/sara
SARA_BUILD_DIR=$HOME/GitHub/DO-CV/sara-build-Release

make -j$(nproc) -C ${SARA_BUILD_DIR}

# # View essential matrix estimation.
# ${SARA_BUILD_DIR}/bin/estimate_essential_matrices \
#   --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
#   --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
#   --read --wait_key

# Triangulation.
${SARA_BUILD_DIR}/bin/triangulate \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --debug
