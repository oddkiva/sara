#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/Basic Operations"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>

#include <fstream>
#include <sstream>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace trt = sara::TensorRT;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<float, N, shakti::PinnedMemoryAllocator>;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_scale_operation)
{
  // List the available GPU devices.
  const auto devices = shakti::get_devices();
  for (const auto& device : devices)
    std::cout << device << std::endl;

  auto cuda_stream = trt::make_cuda_stream();

  auto builder = trt::make_builder();

  // Create a simple scale operation.
  constexpr auto n = 1;
  constexpr auto h = 8;
  constexpr auto w = 8;

  const float scale = 3.f;

  // Instantiate a network and automatically manager its memory.
  auto network = trt::make_network(builder.get());
  {
    SARA_DEBUG << termcolor::green << "Creating the network from scratch!"
               << std::endl;

    // Instantiate an input data.
    auto image_tensor = network->addInput("image", nvinfer1::DataType::kFLOAT,
                                          nvinfer1::Dims3{n, h, w});

    const auto scale_weights = nvinfer1::Weights{
        nvinfer1::DataType::kFLOAT,             //
        reinterpret_cast<const void*>(&scale),  //
        1                                       //
    };

    auto scale_op = network->addScale(  //
        *image_tensor,                  //
        nvinfer1::ScaleMode::kUNIFORM,  //
        {}, scale_weights, {}           //
    );

    // Get the ouput tensor.
    auto image_scaled_tensor = scale_op->getOutput(0);
    network->markOutput(*image_scaled_tensor);
  }

  SARA_DEBUG << termcolor::green
             << "Setting the inference engine with the right configuration!"
             << termcolor::reset << std::endl;

  // Create a inference configuration object.
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),  //
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 32u);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  // If the GPU supports FP16 operations.
  // config->setFlag(nvinfer1::BuilderFlag::kFP16);

  auto plan = trt::HostMemoryUniquePtr{
      builder->buildSerializedNetwork(*network, *config),  //
      trt::host_memory_deleter};
  if (plan.get() == nullptr)
    throw std::runtime_error{"Failed to build TensorRT plan!"};

  auto runtime = trt::RuntimeUniquePtr{
      nvinfer1::createInferRuntime(trt::Logger::instance()),
      &trt::runtime_deleter};

  // Create a CUDA inference engine.
  auto engine = trt::CudaEngineUniquePtr{
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      &trt::engine_deleter};

  // Create a GPU inference context, this is to run the machine learning model
  // on the GPU.
  SARA_DEBUG << termcolor::green << "Setting the inference context!"
             << termcolor::reset << std::endl;
  auto context = trt::ContextUniquePtr{engine->createExecutionContext(),  //
                                       &trt::context_deleter};

  // Create some data and create two GPU device buffers.
  SARA_DEBUG << termcolor::red << "Creating input data and two device buffers!"
             << termcolor::reset << std::endl;
  auto image = PinnedTensor<float, 2>{h, w};
  image.matrix().fill(0.f);
  image(h / 2, w / 2) = 2.f;

  // Inspect the TensorRT log output: there is no padding!
  auto image_convolved = PinnedTensor<float, 2>{{h, w}};
  image_convolved.flat_array().fill(-1.f);

  // Fill some GPU buffers and perform inference.
  SARA_DEBUG << termcolor::green << "Perform inference on GPU!"
             << termcolor::reset << std::endl;

  auto device_buffers = std::vector{
      reinterpret_cast<void*>(image.data()),           //
      reinterpret_cast<void*>(image_convolved.data())  //
  };

  // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
  if (!context->enqueueV2(device_buffers.data(), *cuda_stream, nullptr))
  {
    SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
               << std::endl;
  }

  // Wait for the completion of GPU operations.
  cudaStreamSynchronize(*cuda_stream);

  auto expected_image_output = Eigen::Matrix<float, 8, 8>{};
  expected_image_output.matrix() = image.matrix() * 3;
  std::cout << image_convolved.matrix() << std::endl;

  const auto diff = image_convolved.matrix() - expected_image_output.matrix();
  BOOST_CHECK_LE(diff.norm(), 1e-12f);
}

BOOST_AUTO_TEST_CASE(test_convolution_2d_operation)
{
  // List the available GPU devices.
  const auto devices = shakti::get_devices();
  for (const auto& device : devices)
    std::cout << device << std::endl;

  auto cuda_stream = sara::TensorRT::make_cuda_stream();

  auto builder = sara::TensorRT::make_builder();

  // Create a simple convolution operation.
  static constexpr auto n = 1;
  static constexpr auto h = 8;
  static constexpr auto w = 8;
  static constexpr auto kh = 3;
  static constexpr auto kw = 3;
  static constexpr auto ci = 1;
  static constexpr auto co = 20;

  // Create artificial weights.
  const auto conv1_kernel_weights_vector =
      std::vector<float>(kh * kw * ci * co, 1.f);
  const auto conv1_bias_weights_vector = std::vector<float>(ci * co, 0.f);


  // Instantiate a network and automatically manager its memory.
  auto network = sara::TensorRT::make_network(builder.get());
  {
    SARA_DEBUG << termcolor::green << "Creating the network from scratch!"
               << std::endl;

    // Instantiate an input data.
    auto image_tensor = network->addInput("image", nvinfer1::DataType::kFLOAT,
                                          nvinfer1::Dims4{n, ci, h, w});

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
    const auto conv1_fn = network->addConvolutionNd(
        *image_tensor,
        co,                        // number of filters
        nvinfer1::DimsHW{kh, kw},  // kernel sizes
        conv1_kernel_weights,      // convolution kernel weights
        conv1_bias_weights);       // bias weights

    // Get the ouput tensor.
    auto conv1 = conv1_fn->getOutput(0);
    network->markOutput(*conv1);
  }

  SARA_DEBUG << termcolor::green
             << "Setting the inference engine with the right configuration!"
             << termcolor::reset << std::endl;

  // Create a inference configuration object.
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 32u);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  // If the GPU supports FP16 operations.
  // config->setFlag(nvinfer1::BuilderFlag::kFP16);

  auto plan = trt::HostMemoryUniquePtr{
      builder->buildSerializedNetwork(*network, *config),  //
      trt::host_memory_deleter};
  if (plan.get() == nullptr)
    throw std::runtime_error{"Failed to build TensorRT plan!"};

  auto runtime = trt::RuntimeUniquePtr{
      nvinfer1::createInferRuntime(trt::Logger::instance()),
      &trt::runtime_deleter};

  // Create a CUDA inference engine.
  auto engine = trt::CudaEngineUniquePtr{
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      &trt::engine_deleter};

  // Perform a context to enqueue inference operations in C++.
  SARA_DEBUG << termcolor::green << "Setting the inference context!"
             << termcolor::reset << std::endl;
  auto context = trt::ContextUniquePtr{engine->createExecutionContext(),  //
                                       &trt::context_deleter};

  // Create some data and create two GPU device buffers.
  SARA_DEBUG << termcolor::red << "Creating input data and two device buffers!"
             << termcolor::reset << std::endl;
  auto image = PinnedTensor<float, 2>{h, w};
  image.matrix().fill(0.f);
  image(h / 2, w / 2) = 2.f;

  // Inspect the TensorRT log output: there is no padding!
  auto image_convolved =
      PinnedTensor<float, 3>{{co, h - (kh / 2) * 2, w - (kw / 2) * 2}};
  image_convolved.flat_array().fill(-1.f);

  // Fill some GPU buffers and perform inference.
  SARA_DEBUG << termcolor::green << "Perform inference on GPU!"
             << termcolor::reset << std::endl;

  auto device_buffers = std::vector{
      reinterpret_cast<void*>(image.data()),           //
      reinterpret_cast<void*>(image_convolved.data())  //
  };

  // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
  if (!context->enqueueV2(device_buffers.data(), *cuda_stream, nullptr))
  {
    SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
               << std::endl;
  }

  // Wait for the completion of GPU operations.
  cudaStreamSynchronize(*cuda_stream);

  for (auto c = 0; c < co; ++c)
  {
    SARA_CHECK(c);
    std::cout << image_convolved[c].matrix() << std::endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
