add_subdirectory(third-party)

if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  # Build only a small subset of libraries.
  include(${CMAKE_SOURCE_DIR}/cpp/src/DO/Sara/UseDOSaraCore.cmake)
  include(${CMAKE_SOURCE_DIR}/cpp/src/DO/Sara/UseDOSaraImageIO.cmake)
  add_subdirectory(src/DO/Kalpana/EasyGL)

  # Build the specific Emscripten examples.
  add_subdirectory(examples/Kalpana/Emscripten)
else ()
  add_subdirectory(src/DO)
  add_subdirectory(test)

  if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    return()
  endif ()

  add_subdirectory(examples)
  add_subdirectory(tools)
  add_subdirectory(drafts)
endif ()
