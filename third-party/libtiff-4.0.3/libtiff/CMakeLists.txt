project(libtiff C)

file(GLOB LIBTIFF_SOURCES *.c *.h)
add_library(libjpeg ${LIBTIFF_SOURCES})
