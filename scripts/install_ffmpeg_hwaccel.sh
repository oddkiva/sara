#!/bin/bash
# This script configures and build FFmpeg with CUDA acceleration.
set -ex

sudo apt install -y yasm libx264-dev


GIT_MASTER_REPOSITORY_PATH=${HOME}/GitHub
REPOSITORY_URLS=(
  "https://git.videolan.org/git/ffmpeg/nv-codec-headers.git"
  "hhttps://git.ffmpeg.org/ffmpeg.git"
)

if [ ! -d "${GIT_MASTER_REPOSITORY_PATH}" ]; then
  mkdir -p ${GIT_MASTER_REPOSITORY_PATH}
fi


cd ${GIT_MASTER_REPOSITORY_PATH}

for url in "${REPOSITORY_URLS[@]}"; do
  url_name=$(basename ${url})
  git clone ${url}
done


url_name=$(basename ${REPOSITORY_URLS[0]})
pushd ${GIT_MASTER_REPOSITORY_PATH}/${url_name}
{
  sudo make install
}
popd

url_name=$(basename ${REPOSITORY_URLS[0]})
pushd ${GIT_MASTER_REPOSITORY_PATH}/${url_name}
{
  # Disable stripping for debug information.
  ffmpeg_options="--disable-stripping "

  ffmpeg_options+="--enable-shared "
  ffmpeg_options+="--enable-nonfree  "
  ffmpeg_options+="--enable-gpl "

  # From FFmpeg verions shipped in Ubuntu.
  ffmpeg_options+="--enable-avresample "
  ffmpeg_options+="--enable-avisynth "
  ffmpeg_options+="--enable-gnutls "
  ffmpeg_options+="--enable-ladspa "
  ffmpeg_options+="--enable-libass "
  ffmpeg_options+="--enable-libbluray "
  ffmpeg_options+="--enable-libbs2b "
  ffmpeg_options+="--enable-libcaca "
  ffmpeg_options+="--enable-libcdio "
  ffmpeg_options+="--enable-libflite "
  ffmpeg_options+="--enable-libfontconfig "
  ffmpeg_options+="--enable-libfreetype "
  ffmpeg_options+="--enable-libfribidi "
  ffmpeg_options+="--enable-libgme "
  ffmpeg_options+="--enable-libgsm "
  ffmpeg_options+="--enable-libmp3lame "
  ffmpeg_options+="--enable-libmysofa "
  ffmpeg_options+="--enable-libopencv "
  ffmpeg_options+="--enable-libopenjpeg "
  ffmpeg_options+="--enable-libopenmpt "
  ffmpeg_options+="--enable-libopus "
  ffmpeg_options+="--enable-libpulse "
  ffmpeg_options+="--enable-librubberband "
  ffmpeg_options+="--enable-librsvg "
  ffmpeg_options+="--enable-libshine "
  ffmpeg_options+="--enable-libsnappy "
  ffmpeg_options+="--enable-libsoxr "
  ffmpeg_options+="--enable-libspeex "
  ffmpeg_options+="--enable-libssh "
  ffmpeg_options+="--enable-libtheora "
  ffmpeg_options+="--enable-libtwolame "
  ffmpeg_options+="--enable-libvorbis "
  ffmpeg_options+="--enable-libvpx "
  ffmpeg_options+="--enable-libwavpack "
  ffmpeg_options+="--enable-libwebp "
  ffmpeg_options+="--enable-libx265 "
  ffmpeg_options+="--enable-libxml2 "
  ffmpeg_options+="--enable-libxvid "
  ffmpeg_options+="--enable-libzmq "
  ffmpeg_options+="--enable-libzvbi "
  ffmpeg_options+="--enable-omx "
  ffmpeg_options+="--enable-openal "
  ffmpeg_options+="--enable-opengl "
  ffmpeg_options+="--enable-sdl2 "
  ffmpeg_options+="--enable-libdc1394 "
  ffmpeg_options+="--enable-libdrm "
  ffmpeg_options+="--enable-libiec61883 "
  ffmpeg_options+="--enable-chromaprint "
  ffmpeg_options+="--enable-frei0r "
  ffmpeg_options+="--enable-libx264 "

  # CUDA acceleration.
  ffmpeg_options+="--enable-libnpp "
  ffmpeg_options+="--enable-nvenc "
  ffmpeg_options+="--enable-cuda "
  ffmpeg_options+="--enable-cuda-nvcc "
  ffmpeg_options+="--enable-cuvid "
  ffmpeg_options+="--extra-cflags=-I/usr/local/cuda/include "
  ffmpeg_options+="--extra-ldflags=-L/usr/local/cuda/lib64 "

  ./configure ${ffmpeg_options}
}
popd
