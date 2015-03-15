cmake_minimum_required(VERSION 2.6)

project(DO_VideoIO_tests)

find_package(DO COMPONENTS Core  VideoIO REQUIRED)
include(${DO_USE_FILE})

do_test(VideoIO_Read test_video_io.cpp "${DO_LIBRARIES}" VideoIO)
