cmake_minimum_required(VERSION 3.0)


find_package(DO_Sara COMPONENTS Core Graphics VideoIO REQUIRED)
find_package(OpenMP REQUIRED)


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

function (add_halide_executable source_file)
  get_filename_component(name ${source_file} NAME_WE)
  add_executable(${name} ${source_file})
  target_include_directories(${name}
    PRIVATE ${HALIDE_DISTRIB_DIR}/include
            ${HALIDE_DISTRIB_DIR}/tools)

  target_compile_options(${name} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${name} PRIVATE
    ${HALIDE_LIBRARY}
    ${JPEG_LIBRARIES}
    ${PNG_LIBRARIES}
    ${OpenMP_CXX_FLAGS})

  if (UNIX)
    target_compile_options(${name} PRIVATE
      -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable
      -Wno-missing-field-initializers)
  endif ()

  if (NOT MSVC)
    target_compile_options(${name} PRIVATE -Wno-unknown-pragmas)
  else()
    target_compile_options(${name} PRIVATE /Wd4068)
  endif()
endfunction ()

foreach (lesson ${LESSONS_SOURCE_FILES})
  add_halide_executable(${lesson})
endforeach ()


file(GLOB LESSONS_SOURCE_FILES halide_lesson*.cpp)

foreach (lesson ${LESSONS_SOURCE_FILES})
  get_filename_component(lesson_name ${lesson} NAME_WE)
  add_executable(${lesson_name} ${lesson})
  set_target_properties(${lesson_name} PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_compile_options(${lesson_name} PRIVATE ${OpenMP_CXX_FLAGS})
  target_include_directories(${lesson_name}
    PRIVATE ${HALIDE_DISTRIB_DIR}/include
            ${HALIDE_DISTRIB_DIR}/tools)
  target_compile_options(${lesson_name} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${lesson_name} PRIVATE
    ${DO_Sara_LIBRARIES}
    ${HALIDE_LIBRARY}
    ${OpenMP_CXX_FLAGS})

  if (UNIX)
    target_compile_options(${lesson_name} PRIVATE
      -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable
      -Wno-missing-field-initializers)
  endif ()

  if (NOT MSVC)
    target_compile_options(${lesson_name} PRIVATE -Wno-unknown-pragmas)
  else()
    target_compile_options(${lesson_name} PRIVATE /Wd4068)
  endif()
endforeach()
