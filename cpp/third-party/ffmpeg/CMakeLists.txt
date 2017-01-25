if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Release")
  message("DO = ${DO_Sara_ThirdParty_DIR}")
  install(
    DIRECTORY ${DO_Sara_ThirdParty_DIR}/ffmpeg/include/
    DESTINATION include
    COMPONENT ThirdParty)
  install(
    DIRECTORY ${DO_Sara_ThirdParty_DIR}/ffmpeg/lib/
    DESTINATION lib
    COMPONENT ThirdParty)
  install(
    DIRECTORY ${DO_Sara_ThirdParty_DIR}/ffmpeg/bin/
    DESTINATION bin
    COMPONENT ThirdParty)
endif ()
