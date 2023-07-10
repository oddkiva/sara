if(OnnxRuntime_ROOT)
  list(APPEND OnnxRuntime_SEARCH_PATHS ${OnnxRuntime_ROOT})
endif()

list(APPEND OnnxRuntime_SEARCH_PATHS "/usr")

foreach(search ${OnnxRuntime_SEARCH_PATHS})
  find_path(
    OnnxRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${OnnxRuntime_SEARCH_PATHS}
    PATH_SUFFIXES include)
endforeach()

if(NOT OnnxRuntime_LIBRARY)
  foreach(search ${OnnxRuntime_SEARCH_PATHS})
    find_library(
      OnnxRuntime_LIBRARY
      NAMES onnxruntime
      PATHS ${search}
      PATH_SUFFIXES lib)
  endforeach()
endif()

find_library(
  OnnxRuntime_Providers_CUDA_LIBRARY
  NAMES onnxruntime_providers_cuda
  PATHS ${search}
  PATH_SUFFIXES lib)
find_library(
  OnnxRuntime_Providers_TensorRT_LIBRARY
  NAMES onnxruntime_providers_tensorrt
  PATHS ${search}
  PATH_SUFFIXES lib)
mark_as_advanced(OnnxRuntime_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OnnxRuntime
  REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_Providers_CUDA_LIBRARY
                OnnxRuntime_Providers_TensorRT_LIBRARY OnnxRuntime_INCLUDE_DIR
  VERSION_VAR OnnxRuntime_VERSION_STRING)

if(OnnxRuntime_FOUND)
  set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_INCLUDE_DIR})

  if(NOT OnnxRuntime_LIBRARIES)
    set(OnnxRuntime_LIBRARIES
        ${OnnxRuntime_LIBRARY} ${OnnxRuntime_LIBRARY}
        ${OnnxRuntime_providers_cuda_LIBRARY}
        ${OnnxRuntime_providers_tensorrt_LIBRARY})
  endif()

  add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
  set_target_properties(
    OnnxRuntime::OnnxRuntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                        ${OnnxRuntime_INCLUDE_DIRS})
  set_property(
    TARGET OnnxRuntime::OnnxRuntime
    APPEND
    PROPERTY IMPORTED_LOCATION ${OnnxRuntime_LIBRARY})

  add_library(OnnxRuntime::Providers::CUDA UNKNOWN IMPORTED)
  set_target_properties(
    OnnxRuntime::Providers::CUDA PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                            ${OnnxRuntime_INCLUDE_DIRS})
  set_property(
    TARGET OnnxRuntime::Providers::CUDA
    APPEND
    PROPERTY IMPORTED_LOCATION ${OnnxRuntime_Providers_CUDA_LIBRARY})

  add_library(OnnxRuntime::Providers::TensorRT UNKNOWN IMPORTED)
  set_target_properties(
    OnnxRuntime::Providers::TensorRT PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                ${OnnxRuntime_INCLUDE_DIRS})
  set_property(
    TARGET OnnxRuntime::Providers::TensorRT
    APPEND
    PROPERTY IMPORTED_LOCATION ${OnnxRuntime_Providers_TensorRT_LIBRARY})
endif()
