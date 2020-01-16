cmake_minimum_required(VERSION 3.0)


find_package(DO_Sara COMPONENTS Core Graphics REQUIRED)


set(HALIDE_DISTRIB_DIR /opt/halide)
include("${HALIDE_DISTRIB_DIR}/halide.cmake")

find_library(HALIDE_LIBRARY NAMES Halide
  PATHS ${HALIDE_DISTRIB_DIR}/bin)


file(GLOB LESSONS_SOURCE_FILES
  lesson_01*.cpp
  lesson_02*.cpp
  lesson_03*.cpp
  lesson_04*.cpp
  lesson_05*.cpp
  lesson_06*.cpp
  lesson_07*.cpp
  lesson_08*.cpp
  # lesson_09*.cpp
  # lesson_10*.cpp
  lesson_11*.cpp
  # lesson_12*.cpp
  lesson_13*.cpp
  lesson_14*.cpp
  # lesson_15*.cpp
  # lesson_16*.cpp
  lesson_17*.cpp
  lesson_18*.cpp
  lesson_19*.cpp
  lesson_20*.cpp
  # lesson_21*.cpp
  )

foreach (lesson ${LESSONS_SOURCE_FILES})
  get_filename_component(lesson_name ${lesson} NAME_WE)
  add_executable(${lesson_name} ${lesson})
  target_include_directories(${lesson_name}
    PRIVATE ${HALIDE_DISTRIB_DIR}/include
            ${HALIDE_DISTRIB_DIR}/tools)
  target_link_libraries(${lesson_name}
    PRIVATE ${HALIDE_LIBRARY}
            ${JPEG_LIBRARIES}
            ${PNG_LIBRARIES})
endforeach ()


file(GLOB LESSONS_SOURCE_FILES david_lesson*.cpp)

foreach (lesson ${LESSONS_SOURCE_FILES})
  get_filename_component(lesson_name ${lesson} NAME_WE)
  add_executable(${lesson_name} ${lesson})
  set_target_properties(${lesson_name} PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_include_directories(${lesson_name}
    PRIVATE ${HALIDE_DISTRIB_DIR}/include
            ${HALIDE_DISTRIB_DIR}/tools)
  target_link_libraries(${lesson_name} ${DO_Sara_LIBRARIES} ${HALIDE_LIBRARY})
endforeach()
