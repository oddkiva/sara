find_package(DO_Sara COMPONENTS Core REQUIRED)

find_library(CLBlast_LIBRARY clblast)
if (NOT CLBlast_LIBRARY)
  message(FATAL_ERROR "Could not find CLBlast library!")
endif ()
message("Found CLBlast libraries: ${CLBlast_LIBRARY}")

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

  # Create the library.
  set(DO_Sara_Compute_SOURCE_FILES
    # Apple Accelerate backend.
    Accelerate.hpp
    # CLBlast backend.
    CLBlast.hpp
    # Metal backend.
    MPS.hpp
    MPS/Matrix.hpp MPS/Matrix.mm
    MPS/SGEMMImpl.hpp MPS/SGEMMImpl.mm
    MPS/SGEMM.hpp MPS/SGEMM.cpp)

  set_source_files_properties(${DO_Sara_Compute_SOURCE_FILES}
    PROPERTIES LANGUAGE CXX)

  add_library(DO_Sara_Compute ${DO_Sara_Compute_SOURCE_FILES})

  target_compile_options(DO_Sara_Compute
    PUBLIC -x objective-c++
    PRIVATE -fobjc-arc)

  target_include_directories(DO_Sara_Compute
    PUBLIC ${OpenCL_INCLUDE_DIR})

  target_link_libraries(DO_Sara_Compute
    PUBLIC
    ${DO_Sara_LIBRARIES}
    ${OpenCL_LIBRARY}
    ${CLBlast_LIBRARY}
    ${Accelerate_LIBRARY}
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders")

  set_target_properties(DO_Sara_Compute
    PROPERTIES FOLDER "Libraries/Sara")


  # Add unit tests.
  file(GLOB test_compute_SOURCE_FILES FILES test_*.cpp)

  foreach (file ${test_compute_SOURCE_FILES})
    get_filename_component(filename "${file}" NAME_WE)
    sara_add_test(
      NAME ${filename}
      SOURCES ${file}
      DEPENDENCIES DO_Sara_Compute
      FOLDER Compute)
  endforeach ()
endif ()
