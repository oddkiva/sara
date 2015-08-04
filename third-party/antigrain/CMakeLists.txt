project (antigrain)

cmake_minimum_required(VERSION 2.8)

include_directories (include)
file (GLOB antigrain_HEADERS include/*.h)
file (GLOB antigrain_SOURCES src/*.cpp)
add_library(antigrain STATIC ${antigrain_SOURCES} ${antigrain_HEADERS})
set_target_properties(antigrain
  PROPERTIES
  OUTPUT_NAME_DEBUG antigrain-d)
install(TARGETS antigrain
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT ThirdParty)
