if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fexceptions")
  include(${CMAKE_SOURCE_DIR}/cpp/src/DO/Sara/UseDOSaraCore.cmake)
  add_subdirectory(drafts/OpenCL/GL)
  add_subdirectory(drafts/Emscripten)
  return ()
endif ()

add_subdirectory(third-party)

add_subdirectory(src/DO)

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  return()
endif ()

if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(pipelines)
add_subdirectory(tools)

set(cpp_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(drafts)
