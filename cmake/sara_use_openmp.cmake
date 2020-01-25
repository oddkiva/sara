if (APPLE)
  execute_process(COMMAND brew --prefix libomp
    OUTPUT_VARIABLE OpenMP_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_path(OpenMP_INCLUDE_DIR NAMES omp.h
    PATHS ${OpenMP_DIR}
    PATH_SUFFIXES include
    DOC "OpenMP include directories from brew")

  find_library(OpenMP_LIBRARY NAME omp
    PATHS ${OpenMP_DIR}
    PATH_SUFFIXES lib
    DOC "OpenMP libraries from brew")

  set(OpenMP_LIB_NAMES libomp CACHE STRING "")
  set(OpenMP_FLAGS -Xclang -fopenmp -I${OpenMP_INCLUDE_DIR}
    CACHE STRING "")

  # mark_as_advanced(OpenMP_FOUND OpenMP_INCLUDE_DIR OpenMP_LIBRARY)

  list(APPEND OpenMP_LINKER_FLAGS
    -L${OpenMP_DIR}/lib
    -lomp
    -Wl,-rpath,${OpenMP_DIR}/lib
    CACHE STRING "")

  add_library(OpenMP INTERFACE IMPORTED)
  set_target_properties(OpenMP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${OpenMP_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES ${OpenMP_LIBRARY})
  set(OpenMP_CXX_LIBRARIES OpenMP CACHE STRING "")
else ()
  find_package(OpenMP REQUIRED)
endif ()
