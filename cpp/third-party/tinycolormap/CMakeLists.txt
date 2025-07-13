project(tinycolormap CXX)
set(CMAKE_CXX_STANDARD 11)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/tinycolormap.hpp DESTINATION include)

add_library(tinycolormap INTERFACE)
target_sources(tinycolormap INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/tinycolormap.hpp)
target_include_directories(tinycolormap INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(tinycolormap INTERFACE TINYCOLORMAP_WITH_EIGEN)
