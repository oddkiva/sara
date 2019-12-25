#!/bin/bash
set -ex

GITHUB_MASTER_REPOSITORY_PATH=${HOME}/GitHub
GITHUB_URL=https://github.com

ORGANISATION_NAME=CNugteren
REPOSITORY_NAME=CLBlast
VERSION=1.5.0

if [ ! -d "${GITHUB_MASTER_REPOSITORY_PATH}" ]; then
  mkdir -p ${GITHUB_MASTER_REPOSITORY_PATH}
fi

REPOSITORY_DIR=${GITHUB_MASTER_REPOSITORY_PATH}/${REPOSITORY_NAME}
REPOSITORY_URL=${GITHUB_URL}/${ORGANISATION_NAME}/${REPOSITORY_NAME}

if [ ! -d "${REPOSITORY_DIR}" ]; then
  pushd ${GITHUB_MASTER_REPOSITORY_PATH}
  git clone ${REPOSITORY_URL}
  popd
fi

pushd ${REPOSITORY_DIR}
{
  git fetch origin --prune
  git checkout ${VERSION}

  # Rebuild from scratch
  if [ -d "build" ];then
    rm -rf build
  fi
  mkdir build

  pushd build
  {
    cmake -DTESTS:BOOL=ON -DTUNERS:BOOL=ON ..
    make -j$(nproc)
    # ctest --verbose -j$(nproc)
    sudo make install
  }
  popd
}
popd
