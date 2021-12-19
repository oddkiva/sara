# Treat the header-only library.
shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/Cuda/MultiArray)
shakti_create_common_variables("MultiArray")
add_library(DO_Shakti_MultiArray INTERFACE)
target_sources(DO_Shakti_MultiArray
  INTERFACE "${DO_Shakti_MultiArray_HEADER_FILES}")
target_include_directories(DO_Shakti_MultiArray
  INTERFACE
  $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(DO_Shakti_MultiArray
  INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,AppleClang,Clang>:-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CUDA,GNU,AppleClang,Clang>:-Xcompiler=-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4068>
  $<$<COMPILE_LANG_AND_ID:CUDA,MSVC>:-Xcompiler /wd4068>)
add_library(DO::Shakti::MultiArray ALIAS DO_Shakti_MultiArray)

# Create the reamining libraries.
set(DO_Shakti_LIBRARIES ImageProcessing Segmentation Utilities)
if (NvidiaVideoCodec_ROOT)
  list(APPEND DO_Shakti_LIBRARIES VideoIO)
endif ()
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${CMAKE_CURRENT_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
endforeach ()

target_link_libraries(DO_Shakti_ImageProcessing
  PUBLIC
  DO_Shakti_MultiArray
  DO_Shakti_Utilities)
target_link_libraries(DO_Shakti_Segmentation
  PUBLIC
  DO_Shakti_MultiArray
  DO_Shakti_Utilities)

if (NvidiaVideoCodec_ROOT)
  target_link_libraries(DO_Shakti_VideoIO
    PUBLIC
    ${FFMPEG_LIBRARIES}
    DO::Sara::Core
    nvidia::VideoCodecHelpers)
endif ()
