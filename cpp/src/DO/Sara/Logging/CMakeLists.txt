add_library(DO_Sara_Logging Logger.hpp Logger.cpp)
target_include_directories(DO_Sara_Logging PUBLIC ${CMAKE_SOURCE_DIR}/cpp/src)
target_link_libraries(
  DO_Sara_Logging #
  PUBLIC Boost::log #
         fmt::fmt #
         DO::Sara::Core)
set_target_properties(DO_Sara_Logging PROPERTIES FOLDER "Libraries/Sara")

add_library(DO::Sara::Logging ALIAS DO_Sara_Logging)
