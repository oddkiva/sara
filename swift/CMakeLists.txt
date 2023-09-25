project(Sara-Swift LANGUAGES CXX Swift)

if("${CMAKE_Swift_COMPILER_VERSION}" VERSION_LESS 5.9)
  message(
    FATAL_ERROR
      "Bidirectional C++ Interop requires Swift 5.9 or greater. Have ${CMAKE_Swift_COMPILER_VERSION}"
  )
endif()

# Set up swiftrt.o and runtime library search paths
include(sara_initialize_swift)
# The following provides the function for creating the Swift to C++ bridging
# header
include(sara_add_swift)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(Core)
add_subdirectory(Graphics)
add_subdirectory(ImageProcessing)

add_executable(
  hello_world #
  main.swift Graphics.swift Image.swift ImageIO.swift ImageProcessing.swift
  VideoIO.swift)
target_link_libraries(
  hello_world #
  PRIVATE SaraSwiftCore SaraSwiftGraphics SaraSwiftImageProcessing)
set_target_properties(hello_world #
                      PROPERTIES FOLDER "Swift")
