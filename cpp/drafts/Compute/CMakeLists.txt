find_package(Boost
  COMPONENTS filesystem serialization system unit_test_framework
  REQUIRED)
find_package(DO_Sara COMPONENTS Core REQUIRED)


if (APPLE)
  find_library(Accelerate_LIBRARY Accelerate)
  if (NOT Accelerate_LIBRARY)
    message(FATAL_ERROR "Could not find Accelerate library!")
  endif ()
  message("Found Accelerate framework:\n"
          "\t${Accelerate_LIBRARY}")

  find_library(Foundation_LIBRARY Foundation)
  if (NOT Foundation_LIBRARY)
    message(FATAL_ERROR "Could not find Foundation library!")
  endif ()
  message("Found Foundation framework:\n"
          "\t${Foundation_LIBRARY}")

  find_library(MetalPerformanceShaders_LIBRARY MetalPerformanceShaders)
  if (NOT MetalPerformanceShaders_LIBRARY)
    message(FATAL_ERROR "Could not find MetalPerformanceShaders library!")
  endif ()
  message("Found MetalPerformanceShaders framework:\n"
          "\t${MetalPerformanceShaders_LIBRARY}")

  set(DO_Sara_Compute_SOURCE_FILES
    MPS/Matrix.hpp MPS/Matrix.mm
    MPS/SGEMMImpl.hpp MPS/SGEMMImpl.mm
    MPS/SGEMM.hpp MPS/SGEMM.cpp)
  set_source_files_properties(${DO_Sara_Compute_SOURCE_FILES}
    PROPERTIES LANGUAGE CXX)
  add_library(DO_Sara_Compute ${DO_Sara_Compute_SOURCE_FILES})
  target_compile_options(DO_Sara_Compute PRIVATE -x objective-c++)
  target_link_libraries(DO_Sara_Compute PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders")


  sara_add_test(test_compute_accelerate
    test_compute_accelerate.cpp "${DO_Sara_Libraries}")
  target_include_directories(test_compute_accelerate PRIVATE ${Boost_INCLUDE_DIR})
  target_compile_definitions(test_compute_accelerate PRIVATE -DBOOST_ALL_NO_LIB)
  target_link_libraries(test_compute_accelerate
    ${Boost_LIBRARIES}
    ${Accelerate_LIBRARY})

  sara_add_test(test_compute_mps test_compute_mps.cpp "${DO_Sara_Libraries}")
  target_include_directories(test_compute_mps PRIVATE ${Boost_INCLUDE_DIR})
  target_compile_definitions(test_compute_mps PRIVATE -DBOOST_ALL_NO_LIB)
  target_compile_options(test_compute_mps PRIVATE -x objective-c++)
  target_link_libraries(test_compute_mps
    DO_Sara_Compute
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
    ${Boost_LIBRARIES})
endif ()
