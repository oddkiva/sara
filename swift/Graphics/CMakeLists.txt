find_package(DO_Sara COMPONENTS Graphics ImageIO VideoIO REQUIRED)

add_library(SaraSwiftGraphics CGraphics.cpp)
# This is to find module.modulemap related to CGraphics.hpp
target_include_directories(SaraSwiftGraphics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SaraSwiftGraphics PRIVATE ${DO_Sara_LIBRARIES})
