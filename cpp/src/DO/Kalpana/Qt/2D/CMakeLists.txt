file(GLOB DO_Kalpana_Qt_2D_FILES FILES *.hpp *.cpp)

add_library(DO_Kalpana_Qt_2D ${DO_Kalpana_Qt_2D_FILES})
add_library(DO::Kalpana::Qt::2D ALIAS DO_Kalpana_Qt_2D)

target_link_libraries(
  DO_Kalpana_Qt_2D
  PUBLIC Qt${QT_VERSION}::Core #
         Qt${QT_VERSION}::Widgets #
         Qt${QT_VERSION}::OpenGL #
         $<$<EQUAL:${QT_VERSION},6>:Qt6::OpenGLWidgets> #
         ${OPENGL_LIBRARIES} #
         DO::Sara::Core)
set_property(TARGET DO_Kalpana_Qt_2D PROPERTY FOLDER "Libraries/Kalpana")
