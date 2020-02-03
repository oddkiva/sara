shakti_add_example(
  NAME
  shakti_example_superpixel

  SOURCES
  example_superpixel.cpp

  DEPENDENCIES
  ${DO_Sara_LIBRARIES}
  DO::Shakti::ImageProcessing
  DO::Shakti::Segmentation
  DO::Shakti::Utilities)


# TODO: put this in the Video I/O.
find_library(LIBQUICKTIME_LIBRARY
  NAMES quicktime
  PATHS /usr/lib/x86_64-linux-gnu)

find_path(LIBQUICKTIME_INCLUDE_DIRS
  NAMES lqt/lqt.h
  PATHS /usr/include)

mark_as_advanced(LIBQUICKTIME_LIBRARY LIBQUICKTIME_INCLUDE_DIRS)

if (LIBQUICKTIME_LIBRARY AND LIBQUICKTIME_INCLUDE_DIRS)
  include_directories(${LIBQUICKTIME_INCLUDE_DIRS})

  add_executable(shakti_example_quicktime_video_read
    example_quicktime_video_read.cpp)
  target_link_libraries(example_quicktime_video_read
    ${LIBQUICKTIME_LIBRARY}
    ${DO_Sara_LIBRARIES}
    DO::Shakti::ImageProcessing
    DO::Shakti::Segmentation
    DO::Shakti::Utilities)
endif ()
