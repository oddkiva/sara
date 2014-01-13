cmake_minimum_required(VERSION 2.6)

project(DO_FileSystem_test)

message(STATUS "  - DO_FileSystem_test")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS program_options REQUIRED)

find_package(DO COMPONENTS FileSystem REQUIRED)

include(${DO_USE_FILE})
include_directories(${Boost_INCLUDE_DIRS})

if (MSVC)
  add_definitions(-DBOOST_ALL_DYN_LINK)
endif ()
do_unit_test(FileSystem test.cpp "${DO_LIBRARIES};${Boost_PROGRAM_OPTIONS_LIBRARY}" "FileSystem")