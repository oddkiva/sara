if (NOT OnnxRuntime_FOUND)
  return()
endif ()

add_executable(onnx_runtime_example onnx_runtime_example.cpp)
target_link_libraries(onnx_runtime_example PRIVATE
  OnnxRuntime::OnnxRuntime
  OnnxRuntime::Providers::CUDA
  OnnxRuntime::Providers::TensorRT)
