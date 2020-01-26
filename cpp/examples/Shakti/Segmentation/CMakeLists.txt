shakti_add_example(segmentation_example segmentation_example.cpp)
target_link_libraries(segmentation_example
  PRIVATE ${DO_Sara_LIBRARIES}
          DO_Shakti_ImageProcessing
          DO_Shakti_Segmentation
          DO_Shakti_Utilities
)


find_library(LIBQUICKTIME_LIBRARY
  NAMES quicktime
  PATHS /usr/lib/x86_64-linux-gnu)

find_path(LIBQUICKTIME_INCLUDE_DIRS
  NAMES lqt/lqt.h
  PATHS /usr/include)

mark_as_advanced(LIBQUICKTIME_LIBRARY LIBQUICKTIME_INCLUDE_DIRS)


if (LIBQUICKTIME_LIBRARY AND LIBQUICKTIME_INCLUDE_DIRS)
  include_directories(${LIBQUICKTIME_INCLUDE_DIRS})

  add_executable(quicktime_read_example quicktime_read_example.cpp)
  target_link_libraries(quicktime_read_example
    ${LIBQUICKTIME_LIBRARY}
    ${DO_Sara_LIBRARIES}
    DO_Shakti_ImageProcessing
    DO_Shakti_Segmentation
    DO_Shakti_Utilities)
endif ()
