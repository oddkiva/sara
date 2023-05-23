#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>

#include <numeric>
#include <filesystem>
#include <iostream>
#include <string>


auto main(int const argc, char** const argv) -> int
{
  namespace fs = std::filesystem;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " ONNX_MODEL_PATH\n";
    return 1;
  }

  const auto instance = "task-inference";
  const auto onnx_model_path = fs::path{argv[1]};
  if (!fs::exists(onnx_model_path))
  {
    std::cerr << "ONNX model file does not exist!\n";
    return 1;
  }

  auto env = Ort::Env{OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, instance};

  // Prepare the CUDA backend.
  auto session_options = Ort::SessionOptions{};
  session_options.SetIntraOpNumThreads(1);
  session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Initialize an inference session with CUDA.
  auto session = Ort::Session{
      env,                               //
      onnx_model_path.string().c_str(),  //
      session_options                    //
  };

  auto cuda_memory_info = Ort::MemoryInfo{
      "Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault  //
  };

  auto cuda_allocator = Ort::Allocator{session, cuda_memory_info};

  const auto x_shape = std::array<int64_t, 4>{16, 3, 384, 128};
  const auto x_size =
      std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<int64_t>{});
  auto x_values = std::vector<float>(x_size);
#if 0
  const auto input_name = session.GetInputNameAllocated(0, cuda_allocator);
  std::cout << "input name = " << input_name << std::endl;

  auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_allocator.Alloc(x_size * sizeof(float)),
      CudaMemoryDeleter(&cuda_allocator));
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(),
             cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(
      info_cuda, reinterpret_cast<float*>(input_data.get()), x_size,
      x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {16, 256};
  auto expected_y_size = 16 * 256;
  auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(
      cuda_allocator.Alloc(expected_y_size * sizeof(float)),
      CudaMemoryDeleter(&cuda_allocator));

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(
      info_cuda, reinterpret_cast<float*>(output_data.get()), expected_y_size,
      expected_y_shape.data(), expected_y_shape.size());

  auto binding = Ort::IoBinding{session};
  binding.BindInput("input", bound_x);
  binding.BindOutput("boxes", bound_y);
  binding.BindOutput("classes", bound_y);

  session.Run(Ort::RunOptions(), binding);

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
#endif


  return 0;
}
