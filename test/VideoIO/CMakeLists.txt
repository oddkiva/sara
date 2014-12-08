cmake_minimum_required(VERSION 2.6)

project(DO_VideoIO_tests)

find_package(DO COMPONENTS Core Graphics REQUIRED)
include(${DO_USE_FILE})

# Include FFmpeg header directories.
if (WIN32)
  include_directories(${DO_ThirdParty_DIR}/ffmpeg/include)
  link_directories(${DO_ThirdParty_DIR}/ffmpeg/lib)
endif ()


do_test(VideoIO_Read test_video_io.cpp "${DO_LIBRARIES}" VideoIO)
target_link_libraries(VideoIO_Read avcodec avdevice avformat avutil)