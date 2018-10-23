#!/bin/bash
set -ex


FFMPEG_VERSION="2.8.2"


if [ ! -d "${HOME}/ffmpeg/include" ]; then
  wget http://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2 \
    -O $HOME/ffmpeg-$FFMPEG_VERSION.tar.bz2

  tar xvf $HOME/ffmpeg-$FFMPEG_VERSION.tar.bz2 -C $HOME/

  cd $HOME/ffmpeg-$FFMPEG_VERSION
  {
    ./configure --enable-shared --disable-static --prefix=$HOME/ffmpeg

    make -j`nproc`
    make install
  }
  cd ..
else
  echo "Using cached FFmpeg directory: $HOME/ffmpeg"
fi
