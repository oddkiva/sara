find_package(PkgConfig)

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4)

add_executable(basic_tutorial_1 basic_tutorial_1.cpp)
target_include_directories(basic_tutorial_1 PRIVATE ${GST_INCLUDE_DIRS})
target_link_libraries(basic_tutorial_1 PRIVATE ${GST_LIBRARIES})

add_executable(basic_tutorial_2 basic_tutorial_2.cpp)
target_include_directories(basic_tutorial_2 PRIVATE ${GST_INCLUDE_DIRS})
target_link_libraries(basic_tutorial_2 PRIVATE ${GST_LIBRARIES})
