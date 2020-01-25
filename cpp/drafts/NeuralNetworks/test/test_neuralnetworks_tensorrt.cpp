#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>


auto engine_deleter(nvinfer1::ICudaEngine* engine) -> void
{
  if (engine != nullptr)
    engine->destroy();
  engine = nullptr;
}

using CudaEngineUniquePtr = std::unique_ptr<nvinfer1::ICudaEngine, decltype(&engine_deleter)>;

auto main() -> int
{
  namespace sara = DO::Sara;
  auto builder = sara::TensorRT::make_builder();

  // Instantiate a network and automatically manager its memory.
  auto network = sara::TensorRT::make_network(builder.get());


  // Instantiate an input data.
  auto image_tensor = network->addInput("image", nvinfer1::DataType::kFLOAT,
                                        nvinfer1::Dims3{1, 28, 28});


  // Create artificial weights.
  const auto conv1_kernel_weights_vector =
      std::vector<float>(5 * 5 * 1 * 20, 0.f);
  const auto conv1_bias_weights_vector = std::vector<float>(20, 0.f);


  // Encapsulate the weights using TensorRT data structures.
  const auto conv1_kernel_weights = nvinfer1::Weights{
      nvinfer1::DataType::kFLOAT,
      reinterpret_cast<const void*>(conv1_kernel_weights_vector.data()),
      static_cast<std::int64_t>(conv1_kernel_weights_vector.size())};
  const auto conv1_bias_weights = nvinfer1::Weights{
      nvinfer1::DataType::kFLOAT,
      reinterpret_cast<const void*>(conv1_bias_weights_vector.data()),
      static_cast<std::int64_t>(conv1_bias_weights_vector.size())};


  // Create a convolutional function.
  const auto conv1_fn = network->addConvolution(
      *image_tensor,
      20,                    // number of filters
      {5, 5},                // kernel sizes
      conv1_kernel_weights,  // convolution kernel weights
      conv1_bias_weights);   // bias weight


  // Get the ouput tensor.
  /*  auto conv1 = */ conv1_fn->getOutput(0);


  // Setup the engine.
  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(32);
  builder->allowGPUFallback(true);
  builder->setHalf2Mode(true);


  // Create an engine.
  auto engine = CudaEngineUniquePtr{nullptr, &engine_deleter};

  // Load inference engine from a model weights.
  constexpr auto load_engine = false;
  if constexpr (load_engine)
  {
    auto runtime_deleter = [](nvinfer1::IRuntime* runtime) {
      if (runtime != nullptr)
        runtime->destroy();
      runtime = nullptr;
    };
    auto runtime =
        std::unique_ptr<nvinfer1::IRuntime, decltype(runtime_deleter)>{
            nvinfer1::createInferRuntime(sara::TensorRT::Logger::instance()),
            runtime_deleter};

    auto model_weights_file = std::ifstream{"model_weights.trt"};
    if (!model_weights_file)
      throw std::runtime_error{"Failed to open model weights file!"};

    auto model_weights_stream = std::stringstream{};
    model_weights_stream << model_weights_file.rdbuf();

    model_weights_stream.seekg(0, std::ios::end);
    const auto model_weights_byte_size = model_weights_stream.tellg();

    model_weights_stream.seekg(0, std::ios::beg);
    auto model_weights = std::vector<char>(model_weights_byte_size);
    model_weights_stream.read(model_weights.data(), model_weights.size());

    engine =
        CudaEngineUniquePtr{runtime->deserializeCudaEngine(
                                model_weights.data(), model_weights.size()),
                            &engine_deleter};
  }
  // Build the engine.
  else
  {
    engine = CudaEngineUniquePtr{builder->buildCudaEngine(*network),
                                 &engine_deleter};

    // Save the model weights.
    constexpr auto save_model_weights = false;
    if constexpr (save_model_weights)
    {
      auto model_weights_deleter = [](nvinfer1::IHostMemory* model_stream) {
        if (model_stream != nullptr)
          model_stream->destroy();
        model_stream = nullptr;
      };
      auto model_weights = std::unique_ptr<nvinfer1::IHostMemory,
                                          decltype(model_weights_deleter)>{
          engine->serialize(), model_weights_deleter};

      auto model_weights_stream = std::stringstream{};
      model_weights_stream.seekg(0, model_weights_stream.beg);
      model_weights_stream.write(
          static_cast<const char*>(model_weights->data()),
          model_weights->size());

      auto model_weights_file = std::ofstream{"model_weights.trt"};
      if (!model_weights_file)
        throw std::runtime_error{"Failed to create model weights file!"};
      model_weights_file << model_weights_stream.rdbuf();
    }
  }


  // Perform inference in C++.
  auto context_deleter = [](nvinfer1::IExecutionContext* context) {
    if (context != nullptr)
      context->destroy();
    context = nullptr;
  };
  auto context =
      std::unique_ptr<nvinfer1::IExecutionContext, decltype(context_deleter)>{
          engine->createExecutionContext(), context_deleter};

  // Fill some buffers
  std::vector<void*> device_buffers(2, nullptr);
  context->execute(1, device_buffers.data());

  return 0;
}
