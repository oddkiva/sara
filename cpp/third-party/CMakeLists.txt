sara_step_message("Appending third-party libraries:")

set(THIRDPARTY_FOLDER_NAME "Third-Party")

message(STATUS "  - easyexif")
add_subdirectory(easyexif)
set_property(TARGET easyexif PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

message(STATUS "  - FLANN")
add_subdirectory(flann)
set_property(TARGET flann_cpp_s PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

message(STATUS "  - tinyply")
add_subdirectory(tinyply-2.2)
set_property(TARGET tinyply PROPERTY FOLDER ${THIRDPARTY_FOLDER_NAME})

if (CUDA_FOUND)
  add_subdirectory(nvidia-video-codec-sdk-9.1.23)
  add_subdirectory(gpufilter)
endif ()
