sara_step_message("Appending third-party libraries:")

set(THIRDPARTY_FOLDER_NAME "Third-Party")

message(STATUS "  - fmt")
add_subdirectory(fmt)
set_property(TARGET fmt PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

message(STATUS "  - easyexif")
add_subdirectory(easyexif)

message(STATUS "  - imgui")
add_subdirectory(imgui)

if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
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
