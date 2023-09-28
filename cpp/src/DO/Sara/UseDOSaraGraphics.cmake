if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Graphics_ADDED GLOBAL PROPERTY _DO_Sara_Graphics_INCLUDED)
  if (NOT DO_Sara_Graphics_ADDED)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Graphics)
    sara_create_common_variables("Graphics")
    sara_generate_library("Graphics")

    target_include_directories(DO_Sara_Graphics
      PUBLIC
      ${DO_Sara_INCLUDE_DIR}
      ${DO_Sara_ThirdParty_DIR}
      ${DO_Sara_ThirdParty_DIR}/eigen)

    target_link_libraries(DO_Sara_Graphics
      PUBLIC
      Qt${QT_VERSION}::Widgets
      Qt${QT_VERSION}::OpenGL
      $<$<EQUAL:${QT_VERSION},6>:Qt6::OpenGLWidgets>
      ${OPENGL_LIBRARIES}
      DO::Sara::Core)

    target_compile_definitions(DO_Sara_Graphics
      PUBLIC
      $<$<PLATFORM_ID:Darwin>:GL_SILENCE_DEPRECATION>
      PRIVATE
      $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd26812>
    )

    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  endif()
endif ()
