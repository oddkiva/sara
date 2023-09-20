add_library(SaraSwiftCore CCore.cpp CCore.hpp)

# First ensure that the module.modulemap related to CCore.hpp is accessible to
# swift programs.
target_include_directories(SaraSwiftCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SaraSwiftCore PRIVATE DO::Sara::Core)
set_target_properties(SaraSwiftCore PROPERTIES FOLDER "Swift")

set_target_properties(SaraSwiftCore PROPERTIES Swift_MODULE_NAME "SaraCore")
target_compile_options(
  SaraSwiftCore
  PUBLIC "$<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default>")
