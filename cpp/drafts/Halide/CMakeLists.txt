cmake_minimum_required(VERSION 3.0)


find_package(DO_Sara COMPONENTS Core Graphics VideoIO REQUIRED)
find_package(OpenMP REQUIRED)

if (HALIDE_DISTRIB_DIR)
  find_file(HALIDE_CMAKE_FILEPATH NAMES halide.cmake
    PATHS ${HALIDE_DISTRIB_DIR})

  if (HALIDE_CMAKE_FILEPATH)
    include(${HALIDE_CMAKE_FILEPATH})
  else ()
    message(FATAL_ERROR "Provided HALIDE_DISTRIB_DIR is invalid!")
  endif ()
else ()
  return ()
endif ()

if (WIN32)
  find_library(HALIDE_LIBRARY_DEBUG NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}/Debug)
  find_library(HALIDE_LIBRARY_RELEASE NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}/Release)

  if (NOT HALIDE_LIBRARY_DEBUG OR NOT HALIDE_LIBRARY_RELEASE)
    message(FATAL_ERROR "Could not find Halide libraries")
  endif ()

  set(HALIDE_LIBRARIES
    debug ${HALIDE_LIBRARY_DEBUG}
    optimized ${HALIDE_LIBRARY_RELEASE})

  find_file(HALIDE_DLL_DEBUG NAMES Halide.dll
    PATHS ${HALIDE_DISTRIB_DIR}/Debug)
  find_file(HALIDE_DLL_RELEASE NAMES Halide.dll
    PATHS ${HALIDE_DISTRIB_DIR}/Release)

  if (NOT HALIDE_DLL_DEBUG OR NOT HALIDE_DLL_RELEASE)
    message(FATAL_ERROR "Could not find Halide DLLs")
  endif ()

  set(HALIDE_DLLS
    debug ${HALIDE_DLL_DEBUG}
    optimized ${HALIDE_DLL_RELEASE})

  # Distribute the DLLs in the binary folder.
  if (WIN32)
    file(COPY ${HALIDE_DLL_DEBUG}   DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
    file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)
    file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/RelWithDebInfo)
    file(COPY ${HALIDE_DLL_RELEASE} DESTINATION ${CMAKE_BINARY_DIR}/bin/MinSizeRel)
  endif ()
else ()
  find_library(HALIDE_LIBRARIES NAMES Halide
    PATHS ${HALIDE_DISTRIB_DIR}/bin)

  if (NOT HALIDE_LIBRARIES)
    message(FATAL_ERROR "Could not find Halide libraries")
  endif ()
endif ()


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
    ${HALIDE_LIBRARIES}
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
    target_compile_options(${name} PRIVATE /wd4068)
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
    ${HALIDE_LIBRARIES}
    ${OpenMP_CXX_FLAGS})

  if (UNIX)
    target_compile_options(${lesson_name} PRIVATE
      -Wno-unused-but-set-variable -Wno-unused-parameter -Wno-unused-variable
      -Wno-missing-field-initializers)
  endif ()

  if (NOT MSVC)
    target_compile_options(${lesson_name} PRIVATE -Wno-unknown-pragmas)
  else()
    target_compile_options(${lesson_name} PRIVATE /wd4068)
  endif()
endforeach()
