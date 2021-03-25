cmake_minimum_required(VERSION 2.6)

project(DOImageAnnotator)

find_package(Qt4 REQUIRED)

set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})

set(QtDerived_HEADERS
  MainWindow.hpp
  )

set(QtDerived_SOURCES
  MainWindow.cpp
  )
source_group("Qt Derived Classes" FILES ${QtDerived_HEADERS} ${QtDerived_SOURCES})

qt4_automoc(${QtDerived_SOURCES})
qt4_wrap_cpp(QtDerived_MOC_SOURCES ${QtDerived_HEADERS})
source_group("Moc Files" FILES ${QtDerived_MOC_SOURCES})

set(SOURCES ${QtDerived_MOC_SOURCES} ${QtDerived_HEADERS} ${QtDerived_SOURCES})

add_definitions(-DECPDIR=${DO_DIR}/EvaluateOnECP2011)
#if (WIN32)
#	add_executable(DOImageAnnotator WIN32 main.cpp ${SOURCES})
#else (WIN32)
add_executable(DOImageAnnotator main.cpp ${SOURCES})
#endif (WIN32)
target_link_libraries(DOImageAnnotator ${QT_LIBRARIES})
