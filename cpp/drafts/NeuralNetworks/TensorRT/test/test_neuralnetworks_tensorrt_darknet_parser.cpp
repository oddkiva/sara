#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/Yolo-V4-Tiny"

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


namespace fs = boost::filesystem;

namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace d = sara::Darknet;
namespace trt = sara::TensorRT;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<T, N, shakti::PinnedMemoryAllocator>;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

#if 0
BOOST_AUTO_TEST_CASE(test_yolo_v4_tiny_conversion)
{
  // Instantiate a network and automatically manage its memory.
  auto builder = trt::make_builder();
  auto network = trt::make_network(builder.get());

  // Load the network on the host device (CPU).
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  static const auto yolo_version = 4;
  const auto yolo_model = "yolov" + std::to_string(yolo_version) + "-tiny";
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models" / yolo_model;
  auto hnet = d::load_yolov4_tiny_model(yolov4_tiny_dirpath, yolo_version);

  // Convert the network to TensorRT (GPU).
  auto converter = trt::YoloV4TinyConverter{network.get(), hnet.net};

  // Up until now, I have checked manually that the output of each intermediate
  // layers until max_layers are pretty much equal.
  //
  // TensorRT offers by default convolution, slice, concatenation
  //
  // Upon manual inspection, TensorRT calculates similar results to my CPU
  // implementation:
  // - All layers until layer 31 are correct.
  // - Layers 32,... 37 are correctly implemented.
  //
  // The custom YOLO layer are used for layers 31 and 38 and reports correct
  // results upon manual inspection.
  const auto max_layers = std::numeric_limits<std::size_t>::max();
  converter(max_layers);

  // Create an inference configuration object.
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),  //
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 32u);
  config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
#  ifdef GPU_SUPPORTS_FP16
  // If the GPU supports FP16 operations.
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
#  endif

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

  // This is debug mode.
  if (max_layers != std::numeric_limits<std::size_t>::max())
  {
    // Inspect the TensorRT log output: there is no padding!
    const auto& h_layer = max_layers < hnet.net.size()  //
                              ? *hnet.net[max_layers]
                              : *hnet.net.back();
    const auto& out_sizes = h_layer.output_sizes;
    auto u_out_tensor = PinnedTensor<float, 3>{
        out_sizes(1), out_sizes(2), out_sizes(3)  //
    };

    const auto device_tensors = std::array{
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

    const auto& h_out_tensor = h_layer.output;
    SARA_DEBUG << "Checking layer = " << h_layer.type << "\n"
               << h_layer << std::endl;

    // Check the equality between the CPU implementation and the TensorRT-based
    // network.
    BOOST_CHECK_EQUAL(out_sizes, h_out_tensor.sizes());
    BOOST_CHECK(std::equal(h_out_tensor.begin(), h_out_tensor.end(),
                           u_out_tensor.begin(),
                           [](const float& a, const float& b) {
                             return std::abs(a - b) < 1e-4f;
                           }));

    for (auto i = 0u; i < 2 * 13 * 13; ++i)
    {
      const auto& a = h_out_tensor.data()[i];
      const auto& b = u_out_tensor.data()[i];
      if (std::abs(a - b) > 1e-4f)
        std::cout << sara::format("[OUCH] i=%d me=%f trt=%f\n",  //
                                  i,                             //
                                  h_out_tensor.data()[i],        //
                                  u_out_tensor.data()[i]);
    }
  }
  else
  {
    const auto h_out_tensor =
        std::array{hnet.net[31]->output, hnet.net[38]->output};

    // There are 2 YOLO layers in YOLO v4 Tiny
    auto u_out_tensor = std::array{PinnedTensor<float, 3>{85 * 3, 13, 13},
                                   PinnedTensor<float, 3>{85 * 3, 26, 26}};

    const auto device_tensors = std::array{
        reinterpret_cast<void*>(u_in_tensor.data()),      //
        reinterpret_cast<void*>(u_out_tensor[0].data()),  //
        reinterpret_cast<void*>(u_out_tensor[1].data())   //
    };

    // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
    if (!context->enqueueV2(device_tensors.data(), *cuda_stream, nullptr))
    {
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;
    }

    // Wait for the completion of GPU operations.
    cudaStreamSynchronize(*cuda_stream);

    // Check the equality between the CPU implementation and the TensorRT-based
    // network.
    for (auto i = 0u; i < h_out_tensor.size(); ++i)
      BOOST_CHECK(std::equal(h_out_tensor[i].begin(), h_out_tensor[i].end(),
                             u_out_tensor[i].begin(),
                             [](const float& a, const float& b) {
                               return std::abs(a - b) < 1e-4f;
                             }));

    std::cout << "out 0 =\n" << u_out_tensor[0][0].matrix() << std::endl;
    std::cout << "out 1 =\n" << u_out_tensor[1][0].matrix() << std::endl;
  }
}
#endif

BOOST_AUTO_TEST_CASE(test_yolo_v4_conversion)
{
  // Instantiate a network and automatically manage its memory.
  auto builder = trt::make_builder();
  auto network = trt::make_network(builder.get());

  // Load the network on the host device (CPU).
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  static const auto yolo_version = 4;
  const auto yolo_model = "yolov" + std::to_string(yolo_version);
  const auto yolo_dirpath = data_dir_path / "trained_models" / yolo_model;
  auto hnet = d::load_yolo_model(yolo_dirpath, yolo_version);

  // Convert the network to TensorRT (GPU).
  auto converter = trt::YoloV4TinyConverter{network.get(), hnet.net};

  // Up until now, I have checked manually that the output of each intermediate
  // layers until max_layers are pretty much equal.
  //
  // TensorRT offers by default convolution, slice, concatenation
  //
  // Upon manual inspection, TensorRT calculates similar results to my CPU
  // implementation:
  // - All layers until layer 31 are correct.
  // - Layers 32,... 37 are correctly implemented.
  //
  // The custom YOLO layer are used for layers 31 and 38 and reports correct
  // results upon manual inspection.
  const auto max_layers = 87;  // std::numeric_limits<std::size_t>::max();
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

  // This is debug mode.
  if (max_layers != std::numeric_limits<std::size_t>::max())
  {
    // Inspect the TensorRT log output: there is no padding!
    const auto& h_layer = max_layers < hnet.net.size()  //
                              ? *hnet.net[max_layers]
                              : *hnet.net.back();
    const auto& out_sizes = h_layer.output_sizes;
    auto u_out_tensor = PinnedTensor<float, 3>{
        out_sizes(1), out_sizes(2), out_sizes(3)  //
    };

    const auto device_tensors = std::array{
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

    const auto& h_out_tensor = h_layer.output;
    SARA_DEBUG << "Checking layer [" << max_layers << "] = " << h_layer.type
               << "\n"
               << h_layer << std::endl;

    // Check the equality between the CPU implementation and the TensorRT-based
    // network.
    BOOST_CHECK_EQUAL(out_sizes, h_out_tensor.sizes());
    BOOST_CHECK(std::equal(h_out_tensor.begin(), h_out_tensor.end(),
                           u_out_tensor.begin(),
                           [](const float& a, const float& b) {
                             return std::abs(a - b) < 1e-4f;
                           }));

    for (auto i = 0u; i < 2 * 13 * 13; ++i)
    {
      const auto& a = h_out_tensor.data()[i];
      const auto& b = u_out_tensor.data()[i];
      if (std::abs(a - b) > 1e-4f)
        std::cout << sara::format("[OUCH] i=%d me=%f trt=%f\n",  //
                                  i,                             //
                                  h_out_tensor.data()[i],        //
                                  u_out_tensor.data()[i]);
    }
  }
  else
  {
    const auto h_out_tensor =
        std::array{hnet.net[31]->output, hnet.net[38]->output};

    // There are 3 YOLO layers in YOLO v4
    auto u_out_tensor = std::array{PinnedTensor<float, 3>{85 * 3, 13, 13},
                                   PinnedTensor<float, 3>{85 * 3, 26, 26},
                                   PinnedTensor<float, 3>{85 * 3, 26, 26}};

    const auto device_tensors = std::vector{
        reinterpret_cast<void*>(u_in_tensor.data()),      //
        reinterpret_cast<void*>(u_out_tensor[0].data()),  //
        reinterpret_cast<void*>(u_out_tensor[1].data()),  //
        reinterpret_cast<void*>(u_out_tensor[2].data())   //
    };

    // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
    if (!context->enqueueV2(device_tensors.data(), *cuda_stream, nullptr))
    {
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;
    }

    // Wait for the completion of GPU operations.
    cudaStreamSynchronize(*cuda_stream);

    // Check the equality between the CPU implementation and the TensorRT-based
    // network.
    for (auto i = 0u; i < h_out_tensor.size(); ++i)
      BOOST_CHECK(std::equal(h_out_tensor[i].begin(), h_out_tensor[i].end(),
                             u_out_tensor[i].begin(),
                             [](const float& a, const float& b) {
                               return std::abs(a - b) < 1e-4f;
                             }));

    std::cout << "out 0 =\n" << u_out_tensor[0][0].matrix() << std::endl;
    std::cout << "out 1 =\n" << u_out_tensor[1][0].matrix() << std::endl;
  }
}

BOOST_AUTO_TEST_SUITE_END()
