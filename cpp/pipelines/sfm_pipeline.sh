#!/bin/bash
set -ex

# The dataset.
DATASET_DIR=${HOME}/Desktop/Datasets/sfm/castle_int
RECONSTRUCTION_DATA_FILEPATH=${HOME}/Desktop/Datasets/sfm/castle_int.h5


# Common options.
COMMON_OPTIONS="--dirpath ${DATASET_DIR} "
COMMON_OPTIONS+="--out_h5_file ${RECONSTRUCTION_DATA_FILEPATH} "


# What do you want to do?
for i in "$@"
do
  if [[ "$i" == "detect_sift" ]]; then
    detect_sift ${COMMON_OPTIONS} --overwrite
  fi

  if [[ $i == "view_sift" ]]; then
    detect_sift ${COMMON_OPTIONS} --read
  fi

  if [[ "$i" == "match_keypoints" ]]; then
    match_keypoints ${COMMON_OPTIONS} --overwrite
  fi

  if [[ "$i" == "estimate_fundamental_matrices" ]]; then
    estimate_fundamental_matrices ${COMMON_OPTIONS} --overwrite --debug
  fi

  if [[ "$i" == "inspect_fundamental_matrices" ]]; then
    estimate_fundamental_matrices ${COMMON_OPTIONS} --read --wait_key
  fi

  if [[ "$i" == "estimate_essential_matrices" ]]; then
    estimate_essential_matrices ${COMMON_OPTIONS} --overwrite --debug
  fi

  if [[ "$i" == "inspect_essential_matrices" ]]; then
    estimate_essential_matrices ${COMMON_OPTIONS} --read --wait_key
  fi

  if [[ "$i" == "triangulate" ]]; then
    triangulate ${COMMON_OPTIONS} --debug
  fi
done
