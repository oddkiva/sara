if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_VideoIO_ADDED GLOBAL PROPERTY _DO_Sara_VideoIO_INCLUDED)
  if(NOT DO_Sara_VideoIO_ADDED)
    find_package(SaraFFMPEG REQUIRED)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/VideoIO)
    sara_create_common_variables("VideoIO")
    sara_generate_library("VideoIO")

    if(CMAKE_CUDA_COMPILER)
      target_compile_definitions(DO_Sara_VideoIO PRIVATE HWACCEL)
    endif()
    target_include_directories(DO_Sara_VideoIO PRIVATE ${FFMPEG_INCLUDE_DIRS})
    target_link_libraries(
      DO_Sara_VideoIO
      PRIVATE DO::Sara::Core #
              DO::Sara::ImageProcessing #
              ${FFMPEG_LIBRARIES})
  endif()
endif()
