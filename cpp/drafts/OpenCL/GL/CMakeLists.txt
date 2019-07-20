file(GLOB DO_Sara_OpenCL_GL_FILES FILES *.hpp *.cpp)

add_library(DO_Sara_OpenCL_GL ${DO_Sara_OpenCL_GL_FILES})

target_include_directories(DO_Sara_OpenCL_GL PUBLIC
  ${OpenCL_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR})

target_compile_definitions(DO_Sara_OpenCL_GL PUBLIC
  -DCL_SILENCE_DEPRECATION
  -DGL_SILENCE_DEPRECATION)

target_link_libraries(DO_Sara_OpenCL_GL 
  PRIVATE ${DO_Sara_LIBRARIES}
  PUBLIC ${OpenCL_LIBRARY}
         ${OPENGL_LIBRARIES})

if (NOT APPLE)
  target_link_libraries(DO_Sara_OpenCL_GL PUBLIC ${GLEW_LIBRARIES})
endif ()

set_target_properties(DO_Sara_OpenCL_GL PROPERTIES
  FOLDER "DO Sara Libraries")
