PROJECT (antigrain)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FILE (GLOB antigrain_HEADERS include/*.h)
FILE (GLOB antigrain_SOURCES src/*.cpp)
ADD_LIBRARY (antigrain ${antigrain_SOURCES} ${antigrain_HEADERS})