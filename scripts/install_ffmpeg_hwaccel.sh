#!/bin/bash
#
# This script configures and build FFmpeg with CUDA acceleration.
#
# For more information: https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/
set -ex

sudo apt install -y \
  libbluray-dev \
  libtheora-dev \
  libtwolame-dev \
  libvpx-dev \
  libwebp-dev \
  libx264-dev \
  libx265-dev \
  libxvidcore-dev \
  yasm

GIT_MASTER_REPOSITORY_PATH=${HOME}/GitHub
REPOSITORY_URLS=(
  "https://git.videolan.org/git/ffmpeg/nv-codec-headers.git"
  "https://git.ffmpeg.org/ffmpeg.git"
)

# FFMPEG_VERSION="5.1.2"
FFMPEG_VERSION="4.4.3"  # because of OpenCV on my local machine...


function url_basename()
{
  local url=$1
  local url_name=$(basename ${url})
  local url_basename=${url_name%.*}
  echo ${url_basename}
}

function repo_dirpath()
{
  local url=$1
  echo ${GIT_MASTER_REPOSITORY_PATH}/$(url_basename ${url})
}



if [ ! -d "${GIT_MASTER_REPOSITORY_PATH}" ]; then
  mkdir -p ${GIT_MASTER_REPOSITORY_PATH}
fi


# Clone repositories.
pushd ${GIT_MASTER_REPOSITORY_PATH}
{
  for url in "${REPOSITORY_URLS[@]}"; do
    repo_name=$(url_basename ${url})
    if [ -d "${repo_name}" ]; then
      echo "${repo_name} repository exists, skipping git clone cmd..."
      continue
    else
      echo "Cloning ${url} to ${url_basename}..."
    fi
    git clone ${url}
  done
}
popd


nv_codec_dirpath=$(repo_dirpath ${REPOSITORY_URLS[0]})
pushd ${nv_codec_dirpath}
{
  git pull origin master
  sudo make install
}
popd

# FFmpeg
ffmpeg_dirpath=$(repo_dirpath ${REPOSITORY_URLS[1]})
pushd ${ffmpeg_dirpath}
{
  git clean -fdx
  git fetch origin --prune
  git checkout n${FFMPEG_VERSION}

  # Disable stripping for debug information.
  ffmpeg_options+="--disable-stripping "

  ffmpeg_options+="--enable-shared "
  ffmpeg_options+="--enable-nonfree  "
  ffmpeg_options+="--enable-gpl "

  # From FFmpeg versions shipped in Ubuntu.
  ffmpeg_options+="--enable-avresample "

  # Add the options only if we need them.
  #
  # Decypher these.
  # ffmpeg_options+="--enable-gnutls "
  # ffmpeg_options+="--enable-ladspa "
  # ffmpeg_options+="--enable-libass "
  # ffmpeg_options+="--enable-libbs2b "
  # ffmpeg_options+="--enable-libcaca "
  # ffmpeg_options+="--enable-libcdio "
  # ffmpeg_options+="--enable-libflite "
  # ffmpeg_options+="--enable-libfontconfig "
  # ffmpeg_options+="--enable-libfreetype "
  # ffmpeg_options+="--enable-libfribidi "
  # ffmpeg_options+="--enable-libgme "
  # ffmpeg_options+="--enable-libgsm "
  # ffmpeg_options+="--enable-libmysofa "
  # ffmpeg_options+="--enable-libopencv "
  # ffmpeg_options+="--enable-libopenjpeg "
  # ffmpeg_options+="--enable-libopenmpt "
  # ffmpeg_options+="--enable-libopus "
  # ffmpeg_options+="--enable-librubberband "
  # ffmpeg_options+="--enable-librsvg "
  # ffmpeg_options+="--enable-libsnappy "
  # ffmpeg_options+="--enable-libsoxr "
  # ffmpeg_options+="--enable-libspeex "
  # ffmpeg_options+="--enable-libssh "
  # ffmpeg_options+="--enable-libxml2 "
  # ffmpeg_options+="--enable-libzmq "
  # ffmpeg_options+="--enable-libzvbi "
  # ffmpeg_options+="--enable-omx "
  # ffmpeg_options+="--enable-openal "
  # ffmpeg_options+="--enable-opengl "
  # ffmpeg_options+="--enable-sdl2 "
  # ffmpeg_options+="--enable-libdc1394 "
  # ffmpeg_options+="--enable-libdrm "
  # ffmpeg_options+="--enable-libiec61883 "
  # ffmpeg_options+="--enable-chromaprint "
  # ffmpeg_options+="--enable-frei0r "

  # Audio codecs.
  # ffmpeg_options+="--enable-libvorbis "
  # ffmpeg_options+="--enable-libmp3lame "
  # ffmpeg_options+="--enable-libpulse "
  # ffmpeg_options+="--enable-libshine "
  # ffmpeg_options+="--enable-libwavpack "

  # Video codecs.
  # ffmpeg_options+="--enable-avisynth "
  ffmpeg_options+="--enable-libbluray "
  ffmpeg_options+="--enable-libtheora "
  ffmpeg_options+="--enable-libtwolame "
  ffmpeg_options+="--enable-libvpx "
  ffmpeg_options+="--enable-libwebp "
  ffmpeg_options+="--enable-libx264 "
  ffmpeg_options+="--enable-libx265 "
  ffmpeg_options+="--enable-libxvid "

  # CUDA acceleration.
  ffmpeg_options+="--enable-libnpp "
  ffmpeg_options+="--enable-nvenc "
  ffmpeg_options+="--enable-cuda "
  ffmpeg_options+="--enable-cuda-sdk "
  ffmpeg_options+="--enable-cuvid "
  ffmpeg_options+="--extra-cflags=-I/usr/local/cuda/include "
  ffmpeg_options+="--extra-ldflags=-L/usr/local/cuda/lib64 "

  # If configure fails, that's because we now need to add the following nvccflags.
  # nvidia-smi --query-gpu=compute_cap --format=csv
  #
  # Reference links:
  # https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/
  # https://www.mail-archive.com/ffmpeg-devel@ffmpeg.org/msg120213.html
  ./configure ${ffmpeg_options} --nvccflags="-gencode arch=compute_61,code=sm_61 -O2"
  make -j$(nproc)
  sudo make install
}
popd
