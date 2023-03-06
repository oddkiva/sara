if(TensorRT_ROOT)
  list(APPEND TensorRT_SEARCH_PATHS ${TensorRT_ROOT})
endif()

list(APPEND TensorRT_SEARCH_PATHS "/usr")

find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_SEARCH_PATHS}
  PATH_SUFFIXES include)

find_library(
  TensorRT_LIBRARY
  NAMES nvinfer
  PATHS ${search}
  PATH_SUFFIXES lib)

find_library(
  TensorRT_Plugins_LIBRARY
  NAMES nvinfer_plugin
  PATHS ${TensorRT_SEARCH_PATHS}
  PATH_SUFFIXES lib)

find_library(
  TensorRT_OnnxParser_LIBRARY
  NAMES nvonnxparser
  PATHS ${TensorRT_SEARCH_PATHS}
  PATH_SUFFIXES lib)

mark_as_advanced(TensorRT_INCLUDE_DIR)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MAJOR
       REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MINOR
       REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInfer.h" TensorRT_PATCH
       REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1"
                       TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING
      "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS
    TensorRT_LIBRARY #
    TensorRT_Plugins_LIBRARY #
    TensorRT_OnnxParser_LIBRARY #
    TensorRT_INCLUDE_DIR
  VERSION_VAR TensorRT_VERSION_STRING)

if(TensorRT_FOUND)
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

  if(NOT TensorRT_LIBRARIES)
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY} ${TensorRT_Plugins_LIBRARY})
  endif()

  add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                  ${TensorRT_INCLUDE_DIRS})
  set_property(
    TARGET TensorRT::TensorRT
    APPEND
    PROPERTY IMPORTED_LOCATION ${TensorRT_LIBRARY})

  add_library(TensorRT::Plugins UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::Plugins PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 ${TensorRT_INCLUDE_DIRS})
  set_property(
    TARGET TensorRT::Plugins
    APPEND
    PROPERTY IMPORTED_LOCATION ${TensorRT_Plugins_LIBRARY})

  add_library(TensorRT::OnnxParser UNKNOWN IMPORTED)
  set_target_properties(
    TensorRT::OnnxParser PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 ${TensorRT_INCLUDE_DIRS})
  set_property(
    TARGET TensorRT::OnnxParser
    APPEND
    PROPERTY IMPORTED_LOCATION ${TensorRT_OnnxParser_LIBRARY})
endif()
