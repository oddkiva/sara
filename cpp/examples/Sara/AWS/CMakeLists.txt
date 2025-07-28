if (MSVC)
  string(
    REPLACE ";" "/aws-cpp-sdk-all;"
    SYSTEM_MODULE_PATH "${CMAKE_SYSTEM_PREFIX_PATH}/aws-cpp-sdk-all")
  list(APPEND CMAKE_PREFIX_PATH ${SYSTEM_MODULE_PATH})
endif ()

list(APPEND SERVICE_COMPONENTS ec2 kinesisvideo kinesis-video-archived-media)

find_package(AWSSDK COMPONENTS ${SERVICE_COMPONENTS})
find_package(DO_Sara COMPONENTS Core Graphics VideoIO REQUIRED)

if (NOT AWSSDK_FOUND)
  return ()
endif ()


if (MSVC AND AWSSDK_INSTALL_AS_SHARED_LIBS)
  # Copy relevant AWS SDK for C++ libraries into the current binary directory
  # for running and debugging.

  # If you are building from the command line, you may need to uncomment this
  # and set the proper subdirectory to the executables' location.
  # set(BIN_SUB_DIR "/Debug")

  AWSSDK_CPY_DYN_LIBS(SERVICE_COMPONENTS "" ${CMAKE_CURRENT_BINARY_DIR}${BIN_SUB_DIR})
endif ()

add_executable(hello_kvs hello_kvs.cpp)
set_target_properties(hello_kvs PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS})
target_link_libraries(hello_kvs
  ${AWSSDK_LINK_LIBRARIES}
  ${DO_Sara_LIBRARIES})
