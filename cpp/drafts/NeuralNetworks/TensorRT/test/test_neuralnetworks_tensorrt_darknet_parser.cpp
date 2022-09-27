#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>

#include <drafts/NeuralNetworks/Darknet/Parser.hpp>
#include <drafts/NeuralNetworks/TensorRT/DarknetParser.hpp>
#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>

#include <thread>


namespace fs = boost::filesystem;

namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace d = sara::Darknet;
namespace trt = sara::TensorRT;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<float, N, shakti::PinnedMemoryAllocator>;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_yolo_v4_tiny_conversion)
{
  // Instantiate a network and automatically manager its memory.
  auto builder = trt::make_builder();
  auto network = trt::make_network(builder.get());

  // Load the network on the host device (CPU).
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";
  auto hnet = d::load_yolov4_tiny_model(yolov4_tiny_dirpath);

  // Convert the network to TensorRT (GPU).
  auto converter = trt::YoloV4TinyConverter{network.get(), hnet.net};

  // Up until now, I have checked manually that the output of each intermediate
  // layers until max_layers are pretty much equal.
  //
  // Until I implement YOLO correctly, this will fail
  // - Everything is fine until layer 30
  // - Layers 31, 32,... 37 are correctly implemented.
  // - Layer 31 and 38 are the yolo layers, which still fails
  const auto max_layers = 31;  // std::numeric_limits<std::size_t>::max();
  converter(max_layers);


  // Create an inference configuration object.
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),  //
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 32u);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
#ifdef GPU_SUPPORTS_FP16
  // If the GPU supports FP16 operations.
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
#endif

  // Serialize the network definition and weights for TensorRT.
  auto plan = trt::HostMemoryUniquePtr{
      builder->buildSerializedNetwork(*network, *config),  //
      trt::host_memory_deleter};
  if (plan.get() == nullptr)
    throw std::runtime_error{"Failed to build TensorRT plan!"};


  // Create a runtime.
  auto runtime = trt::RuntimeUniquePtr{
      nvinfer1::createInferRuntime(trt::Logger::instance()),
      &trt::runtime_deleter};

  // Create or load an engine.
  auto engine = trt::CudaEngineUniquePtr{nullptr, &trt::engine_deleter};
  engine = trt::CudaEngineUniquePtr{
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      &trt::engine_deleter};


  // Create an inference context.
  SARA_DEBUG << termcolor::green << "Setting the inference context!"
             << termcolor::reset << std::endl;
  auto cuda_stream = trt::make_cuda_stream();
  auto context = trt::ContextUniquePtr{engine->createExecutionContext(),  //
                                       &trt::context_deleter};

  // Prepare the input tensor
  const auto image = sara::imread<sara::Rgb8>(src_path("data/dog.jpg"));

  // Resize the image to the network input sizes.
  const auto& input_layer =
      dynamic_cast<const sara::Darknet::Input&>(*hnet.net.front());
  const auto image_resized =
      sara::resize(image, {input_layer.width(), input_layer.height()})
          .convert<sara::Rgb32f>();
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});

  hnet.forward(image_tensor);

  // Resize the host tensor.
  auto u_in_tensor = PinnedTensor<float, 3>{};
  u_in_tensor.resize(3, input_layer.height(), input_layer.width());
  std::copy(image_tensor.begin(), image_tensor.end(), u_in_tensor.begin());

  // Inspect the TensorRT log output: there is no padding!
  const auto& out_sizes = hnet.net[max_layers]->output_sizes;
  auto u_out_tensor = PinnedTensor<float, 3>{
      out_sizes(1), out_sizes(2), out_sizes(3)  //
  };

  auto device_tensors = std::vector{
      reinterpret_cast<void*>(u_in_tensor.data()),  //
      reinterpret_cast<void*>(u_out_tensor.data())  //
  };

  // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
  if (!context->enqueueV2(device_tensors.data(), *cuda_stream, nullptr))
  {
    SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
               << std::endl;
  }

  // Wait for the completion of GPU operations.
  cudaStreamSynchronize(*cuda_stream);

  const auto& h_layer = *hnet.net[max_layers];
  const auto& h_out_tensor = h_layer.output;

  SARA_DEBUG << "Checking layer = " << h_layer.type << "\n"
             << h_layer << std::endl;

  // Check the equality between the CPU implementation and the TensorRT-based
  // network.
  BOOST_CHECK_EQUAL(out_sizes, h_out_tensor.sizes());
  BOOST_CHECK(std::equal(
      h_out_tensor.begin(), h_out_tensor.end(), u_out_tensor.begin(),
      [](const float& a, const float& b) { return std::abs(a - b) < 1e-4f; }));
}

BOOST_AUTO_TEST_SUITE_END()
