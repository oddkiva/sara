CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#README
# Compile only daisy as a library.
# do not compile the binary because third party libraries are required
# such as png zlib and jpeg

project(daisy)

IF (UNIX OR APPLE)
  check_symbol_exists(mmap64 "sys/mman.h" HAVE_MMAP64)
ENDIF ()

if(${HAVE_MMAP64})
    message("System has large file support.")
    add_definitions(-DHAVE_LARGE_FILE_SUPPORT)
else()
    message("System hasn't large file support.")
endif()

# Detect OpenMP
if (NOT APPLE)
  find_package(OpenMP)
  if (OPENMP_FOUND) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    option(USE_OPENMP "Use OpenMP for parallelization" ON)
    add_definitions(-DUSE_OPENMP)
  else(OPENMP_FOUND)
    option(USE_OPENMP "Use OpenMP for parallelization" OFF)
  endif (OPENMP_FOUND)
endif ()

include_directories(include)

# Make the headers appear in IDEs.
file(GLOB daisy_hdrs include/kutility/*.h include/daisy/*.h)

set(daisy_srcs
    src/daisy.cpp
    src/corecv.cpp
    src/image_io_bmp.cpp
    src/image_io_pnm.cpp
    src/image_io_png.cpp
    src/image_io_jpeg.cpp
    src/image_manipulation.cpp
    src/progress_bar.cpp
    src/general.cpp
    src/interaction.cpp
)

add_library(Tola_DAISY ${daisy_srcs} ${daisy_hdrs})

find_package(DO COMPONENTS Graphics Features FeatureDetectors REQUIRED)
include(${DO_USE_FILE})

add_definitions(-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR})
add_executable(Tola_DAISY_main src/main.cpp)
do_use_modules(Tola_DAISY_main "Tola_DAISY;${DO_LIBRARIES}")

