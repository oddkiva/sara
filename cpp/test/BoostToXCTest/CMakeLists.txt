add_library(BoostToXCTest BoostToXCTest.mm)

# target_compile_options(BoostToXCTest
#   PUBLIC -x objective-c++
#   PRIVATE -fobjc-arc)

target_include_directories(BoostToXCTest
  PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${XCTest_INCLUDE_DIRS})

target_link_libraries(BoostToXCTest
  PUBLIC
  "-framework Foundation"
  ${XCTest_LIBRARIES})

