file(GLOB DO_Kalpana_EasyGL_FILES FILES *.hpp *.cpp)

add_library(DO_Kalpana_EasyGL ${DO_Kalpana_EasyGL_FILES})

target_include_directories(DO_Kalpana_EasyGL PUBLIC ${GLEW_INCLUDE_DIR})
target_compile_definitions(DO_Kalpana_EasyGL PUBLIC GL_SILENCE_DEPRECATION)
target_link_libraries(
  DO_Kalpana_EasyGL
  PRIVATE DO::Sara::Core
  PUBLIC ${OPENGL_LIBRARIES} #
         # N.B.: do not link with ${GLEW_LIBRARIES}: it will not work on Windows!
         $<$<NOT:$<PLATFORM_ID:Apple>>:GLEW::GLEW>)

set_target_properties(DO_Kalpana_EasyGL PROPERTIES FOLDER "Libraries/Kalpana")
add_library(DO::Kalpana::EasyGL ALIAS DO_Kalpana_EasyGL)