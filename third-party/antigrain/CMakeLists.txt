project (antigrain)

cmake_minimum_required(VERSION 2.8)

include_directories (include)
file (GLOB antigrain_HEADERS include/*.h)
file (GLOB antigrain_SOURCES src/*.cpp)
add_library (antigrain ${antigrain_SOURCES} ${antigrain_HEADERS})
install(TARGETS antigrain
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT ThirdParty)
