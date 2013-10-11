project(libjpeg C)

file(GLOB LIBJPEG_SOURCES *.c)

file(GLOB LIBJPEG_HEADERS *.h)
file(GLOB LIBJPEG_ASM *.asm)
add_library(libjpeg ${LIBJPEG_SOURCES} ${LIBJPEG_HEADERS} ${LIBJPEG_ASM})
