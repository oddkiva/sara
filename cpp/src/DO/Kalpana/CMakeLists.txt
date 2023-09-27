if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  return ()
endif ()

add_subdirectory(EasyGL)
add_subdirectory(Qt)
# add_subdirectory(Shape)
