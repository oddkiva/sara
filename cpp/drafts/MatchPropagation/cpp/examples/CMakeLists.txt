find_package(DO_Sara COMPONENTS 
  Core Graphics KDTree Geometry FileSystem Features FeatureDetectors
  FeatureDescriptors Match FeatureMatching REQUIRED)

add_definitions(-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR})
add_executable(DO_GrowRegion_test main.cpp)
do_use_modules(DO_GrowRegion_test "${DO_LIBRARIES};DO_GrowRegion")
set_property(TARGET DO_GrowRegion_test PROPERTY FOLDER "DO 4th-Order Match Propagation Test")
