sara_step_message("Appending third-party libraries:")

set(THIRDPARTY_FOLDER_NAME "Third-Party")

message(STATUS "  - easyexif")
add_subdirectory(easyexif)
set_property(TARGET easyexif PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

if (CMAKE_SYSTEM_NAME STREQUAL EMSCRIPTEN)
  return ()
endif ()

message(STATUS "  - FLANN")
add_subdirectory(flann)
set_property(TARGET flann_cpp_s PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

message(STATUS "  - termcolor")
add_subdirectory(termcolor)

message(STATUS "  - tinyply")
add_subdirectory(tinyply-2.2)
set_property(TARGET tinyply PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

if (CMAKE_CUDA_COMPILER)
  if (NvidiaVideoCodec_ROOT)
    add_subdirectory(nvidia-video-codec-sdk-9.1.23)
  endif ()
endif ()
