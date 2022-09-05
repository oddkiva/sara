file(GLOB DO_Kalpana_Qt_3D_FILES FILES *.hpp *.cpp)

add_library(DO_Kalpana_Qt_3D ${DO_Kalpana_Qt_3D_FILES})
add_library(DO::Kalpana::Qt::3D ALIAS DO_Kalpana_Qt_3D)

target_link_libraries(
  DO_Kalpana_Qt_3D
  PUBLIC Qt${QT_VERSION}::Core #
         Qt${QT_VERSION}::Widgets #
         Qt${QT_VERSION}::OpenGL #
         $<$<EQUAL:${QT_VERSION},6>:Qt6::OpenGLWidgets> #
         ${OPENGL_LIBRARIES} #
         DO::Sara::Core)
