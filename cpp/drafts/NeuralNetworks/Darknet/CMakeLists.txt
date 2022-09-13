file(GLOB DO_Sara_Darknet_SOURCE_FILES FILES *.hpp *.cpp)

add_library(DO_Sara_Darknet ${DO_Sara_Darknet_SOURCE_FILES})
add_library(DO::Sara::Darknet ALIAS DO_Sara_Darknet)

target_link_libraries(DO_Sara_Darknet PUBLIC DO::Sara::Core
                                             DO::Sara::ImageProcessing
                                             Boost::filesystem)

add_subdirectory(test)
add_subdirectory(examples)
