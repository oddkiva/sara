#!/bin/bash

set -e

# Install FFmpeg.
if [ ! -d "$HOME/ffmpeg/include" ]; then
  FFMPEG_VERSION="2.8.2"
  wget http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 \
    -O $HOME/ffmpeg-$FFMPEG_VERSION.tar.bz2
  tar xvf $HOME/ffmpeg-$FFMPEG_VERSION.tar.bz2 -C $HOME/
  cd $HOME/ffmpeg-$FFMPEG_VERSION
  ./configure --enable-shared --disable-static --prefix=$HOME/ffmpeg
  make -j`nproc`
  make install
  cd ..
else
  echo "Using cached FFmpeg directory: $HOME/ffmpeg"
fi

# Install lcov for code coverage.
if [ ! -d "$HOME/lcov/bin" ]; then
  LCOV_VERSION="1.12"
  wget http://downloads.sourceforge.net/ltp/lcov-$LCOV_VERSION.tar.gz -O $HOME/lcov.tar.gz
  tar -xzf $HOME/lcov.tar.gz -C $HOME/lcov --strip-components=1
  gem install coveralls-lcov
else
  echo "Using cached lcov directory: $HOME/lcov"
fi

# Use lcov in the cached directory.
export PATH=$HOME/lcov/bin:$PATH
