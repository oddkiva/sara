find_package(DO_Sara COMPONENTS Core REQUIRED)

add_library(SaraSwiftCore CCore.cpp)
# First ensure that the module.modulemap related to CCore.hpp is accessible to
# swift programs.
target_include_directories(SaraSwiftCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SaraSwiftCore PRIVATE ${DO_Sara_LIBRARIES})
set_target_properties(SaraSwiftCore PROPERTIES
  FOLDER "Swift")
