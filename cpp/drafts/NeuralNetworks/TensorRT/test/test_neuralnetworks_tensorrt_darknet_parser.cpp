#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/Yolo-V4"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>

#include <drafts/NeuralNetworks/Darknet/Debug.hpp>
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


auto get_yolov4_model() -> d::Network
{
  // Load the network on the host device (CPU).
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  static const auto yolo_version = 4;
  const auto yolo_model = "yolov" + std::to_string(yolo_version);
  const auto yolo_dirpath = data_dir_path / "trained_models" / yolo_model;
  auto hnet = d::load_yolo_model(yolo_dirpath, yolo_version, false);
  return hnet;
}

auto get_yolov4_intermediate_outputs() -> std::vector<sara::Tensor_<float, 4>>
{
  const auto yolov4_intermediate_output_dir =
      "/home/david/GitHub/darknet/yolov4";
  const auto gt =
      d::read_all_intermediate_outputs(yolov4_intermediate_output_dir);
  return gt;
}

auto get_image_tensor(const d::Network& hnet) -> sara::Tensor_<float, 4>
{
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
  SARA_CHECK(image_tensor.sizes().transpose());

  return image_tensor;
}

// Sweet this works...
BOOST_AUTO_TEST_CASE(test_yolo_v4_check_each_unary_layer_individually)
{
  // Get my CPU inference implementation of YOLO v4.
  auto hnet = get_yolov4_model();
  hnet.debug = true;

  // The ground-truth test data.
  const auto gt = get_yolov4_intermediate_outputs();

  // Instantiate a single CUDA stream for everything.
  auto cuda_stream = trt::make_cuda_stream();

#if defined(TEST_ALL_LAYERS)
  for (auto layer_idx = 2u; /* not from 1u because I haven't fetched the input
                              image tensor yet */
       layer_idx < hnet.net.size(); ++layer_idx)
#else
  const auto layer_idx = 87u;
#endif
  {
    const auto& test_in_data = gt[layer_idx - 2];
    const auto& test_out_data = gt[layer_idx - 1];

    // Get the host tensors.
    auto h_in_tensor = hnet.get_input(layer_idx);
    auto h_out_tensor = hnet.get_output(layer_idx);

    // Create and initialize the CUDA tensors.
    auto u_in_tensor = PinnedTensor<float, 3>{h_in_tensor.sizes().tail(3)};
    auto u_out_tensor = PinnedTensor<float, 3>{h_out_tensor.sizes().tail(3)};
    h_in_tensor = test_in_data;
    u_in_tensor = test_in_data[0];

    // For now, we only check layers that accepts only one input tensor.
    SARA_DEBUG << "Forwarding data to CPU inference implementation...\n";
    if (auto layer = dynamic_cast<d::Convolution*>(hnet.net[layer_idx].get()))
      layer->forward(h_in_tensor);
    else if (auto layer = dynamic_cast<d::MaxPool*>(hnet.net[layer_idx].get()))
      layer->forward(h_in_tensor);
    else if (auto layer = dynamic_cast<d::Upsample*>(hnet.net[layer_idx].get()))
      layer->forward(h_in_tensor);
    else if (auto layer = dynamic_cast<d::Yolo*>(hnet.net[layer_idx].get()))
      layer->forward(h_in_tensor);
    else
    {
      SARA_DEBUG << "SKIPPING THIS POSSIBLY NON-UNARY LAYER... (BUILD FROM END "
                    "TO END INSTEAD...)\n";
#if defined(TEST_ALL_LAYERS)
      continue;
#else
      return;
#endif
    }

    // Build the mini-network consisting of only the convolution layer.
    auto net_builder = trt::make_builder();
    auto net = trt::make_network(net_builder.get());
    auto converter = trt::YoloV4Converter{net.get(), hnet.net};
    converter(layer_idx);

    // Serialize the TensorRT engine
    const auto plan = trt::serialize_network_into_plan(net_builder, net,  //
                                                       false /* use_fp16*/);

    // Create a TensorRT runtime.
    auto runtime = trt::RuntimeUniquePtr{
        nvinfer1::createInferRuntime(trt::Logger::instance()),
        &trt::runtime_deleter};

    // Create or load an TensorRT engine.
    auto engine = trt::CudaEngineUniquePtr{nullptr, &trt::engine_deleter};
    engine = trt::CudaEngineUniquePtr{
        runtime->deserializeCudaEngine(plan->data(), plan->size()),
        &trt::engine_deleter};

    // Create a TensorRT inference context.
    auto context = trt::ContextUniquePtr{engine->createExecutionContext(),  //
                                         &trt::context_deleter};

    h_in_tensor = test_in_data;

    BOOST_CHECK(std::equal(h_out_tensor.begin(), h_out_tensor.end(),
                           test_out_data.begin(),
                           [](const float& a, const float& b) {
                             return std::abs(a - b) < 1e-4f;
                           }));

    // TensorRT implementation.
    SARA_DEBUG << "Forwarding data to TensorRT implementation...\n";
    const auto device_tensors = std::array{
        reinterpret_cast<void*>(u_in_tensor.data()),  //
        reinterpret_cast<void*>(u_out_tensor.data())  //
    };
    if (!context->enqueueV2(device_tensors.data(), *cuda_stream, nullptr))
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;

    // Wait for the completion of GPU operations.
    cudaStreamSynchronize(*cuda_stream);

    SARA_DEBUG << "Checking output of layer [" << layer_idx
               << "] = " << hnet.net[layer_idx]->type << "\n"
               << *hnet.net[layer_idx] << std::endl;

    // Check the equality between the CPU implementation and the
    // TensorRT-based network.
    BOOST_REQUIRE_EQUAL(h_out_tensor.sizes().tail(3), u_out_tensor.sizes());

    static constexpr auto thresh = 1e-4f;
    // Check a little bit of the output tensors.
    auto num_errors = 0;
    for (auto i = 0u; i < u_out_tensor.size(); ++i)
    {
      const auto& a = h_out_tensor.data()[i];
      const auto& b = u_out_tensor.data()[i];
      if (std::abs(a - b) > thresh)
      {
        std::cout << sara::format("[OUCH] i=%d me=%f trt=%f\n",  //
                                  i,                             //
                                  h_out_tensor.data()[i],        //
                                  u_out_tensor.data()[i]);
        ++num_errors;
      }
      if (num_errors > 20)
        break;
    }

    SARA_CHECK(u_in_tensor.data());
    SARA_CHECK(u_out_tensor.data());

    // The full check.
    BOOST_REQUIRE(std::equal(h_out_tensor.begin(), h_out_tensor.end(),
                             u_out_tensor.begin(),
                             [](const float& a, const float& b) {
                               return std::abs(a - b) < thresh;
                             }));
  }
}


BOOST_AUTO_TEST_CASE(test_yolo_v4_conversion_incrementally_and_exhaustively)
{
  // Get my CPU inference implementation of YOLO v4.
  auto hnet = get_yolov4_model();
  hnet.debug = true;

  // Read a dog image.
  const auto image_tensor = get_image_tensor(hnet);


  // Make a unique CUDA stream.
  auto cuda_stream = trt::make_cuda_stream();

  // Copy the host tensor to the input CUDA tensor.
  auto u_in_tensor =
      PinnedTensor<float, 3>{3, image_tensor.size(2), image_tensor.size(3)};
  u_in_tensor = image_tensor[0];
  BOOST_REQUIRE(std::equal(u_in_tensor.begin(), u_in_tensor.end(),  //
                           image_tensor.begin()));

  // Verify the network conversion to TensorRT incrementally and exhaustively.
  //
  // Everything goes well until layer 87...
  // for (auto max_layers = 88u; max_layers < hnet.net.size(); ++max_layers)
  auto max_layers = 35u;

// layers = 35
//
// h_out_tensor
//  -0.239912  -0.276882  -0.112395  -0.306818  -0.248383  -0.154634  -0.184812  -0.122076
//  -0.306911   0.204397  -0.221075     1.5993   0.781725  -0.191988  -0.164775  -0.303083
//  -0.278409   0.774572  -0.216393 -0.0724351   0.490605  -0.308843  -0.295758  -0.252452
//  0.0324171 -0.0383892  -0.279219  -0.279822    0.33445  -0.229523  -0.233142  -0.186258
//  -0.289081  -0.298239  -0.307617  -0.305277    1.39844   -0.22397  -0.129098  -0.255643
//   -0.30245  -0.196463  -0.302754  -0.249703    1.64081  -0.243826   0.118683  -0.306845
//  -0.231999   0.768396  -0.301371  0.0713109   0.402886  -0.308842  -0.236937  -0.241136
//   -0.30534    1.60836  -0.308838   -0.30884   0.549651  -0.205581  -0.194776  -0.308751
// u_out_tensor
//  -0.289367  -0.264936  -0.101996   -0.30311   -0.25898  -0.118093  -0.163118 -0.0736392
//   -0.26204  -0.126781  -0.286619    1.04209    1.06169  -0.284401  -0.295792  -0.159279
//  -0.302032  0.0600801  -0.290587  0.0266602    1.20102  -0.289216  -0.299499  -0.175496
//  -0.198176  -0.258859  -0.243346  -0.307519   0.597543  -0.304013  -0.290579  -0.257453
//  -0.228253  -0.238514   -0.29665  -0.298039    2.14789  -0.131796  -0.163164  -0.232221
//  -0.278594  -0.268113   0.549058   -0.26297    2.43282  -0.195115  -0.180264  -0.291472
// -0.0675805    1.79673   0.876684    2.31913    2.55072   0.498026  -0.308815  -0.307724
//  -0.291981    2.37784   0.766488  -0.165548    1.24549    1.00571  -0.281311  -0.305154
  {
    // Build the mini-network consisting of only the convolution layer.
    auto net_builder = trt::make_builder();
    auto net = trt::make_network(net_builder.get());
    auto converter = trt::YoloV4Converter{net.get(), hnet.net};
    converter(1, max_layers + 1);

    // Serialize the TensorRT engine
    const auto plan = trt::serialize_network_into_plan(net_builder, net,  //
                                                       false /* use_fp16*/);

    // Create a TensorRT runtime.
    auto runtime = trt::RuntimeUniquePtr{
        nvinfer1::createInferRuntime(trt::Logger::instance()),
        &trt::runtime_deleter};

    // Create or load an TensorRT engine.
    auto engine = trt::CudaEngineUniquePtr{nullptr, &trt::engine_deleter};
    engine = trt::CudaEngineUniquePtr{
        runtime->deserializeCudaEngine(plan->data(), plan->size()),
        &trt::engine_deleter};

    // Create a TensorRT inference context.
    SARA_DEBUG << termcolor::green << "Setting the inference context!"
               << termcolor::reset << std::endl;
    auto context = trt::ContextUniquePtr{engine->createExecutionContext(),  //
                                         &trt::context_deleter};

    SARA_DEBUG << "Forwarding data to CPU inference implementation...\n";
    hnet.forward(image_tensor, max_layers);

    // Inspect the TensorRT log output: there is no padding!
    const auto& h_layer = *hnet.net[max_layers];
    const auto& h_out_sizes = h_layer.output_sizes;
    auto u_out_tensor = PinnedTensor<float, 3>{
        h_out_sizes(1), h_out_sizes(2), h_out_sizes(3)  //
    };
    u_out_tensor.flat_array().fill(0);
    SHAKTI_SYNCHRONIZED_CHECK();

    const auto device_tensors = std::array{
        reinterpret_cast<void*>(u_in_tensor.data()),  //
        reinterpret_cast<void*>(u_out_tensor.data())  //
    };

    // Enqueue the CPU pinned <-> GPU tranfers and the convolution task.
    SARA_DEBUG << "Forwarding data to TensorRT implementation...\n";

    // Instantiate a single CUDA stream for everything.
    if (!context->enqueueV2(device_tensors.data(), *cuda_stream, nullptr))
      SARA_DEBUG << termcolor::red << "Execution failed!" << termcolor::reset
                 << std::endl;
    cudaStreamSynchronize(*cuda_stream);
    SHAKTI_SYNCHRONIZED_CHECK();

    const auto& h_out_tensor = h_layer.output;
    SARA_DEBUG << "Checking layer [" << max_layers << "] = " << h_layer.type
               << "\n"
               << h_layer << std::endl;

    // Check the equality between the CPU implementation and the
    // TensorRT-based network.
    BOOST_REQUIRE_EQUAL(u_out_tensor.sizes(), h_out_tensor.sizes().tail(3));

    // Check a little bit of the output tensors.
    static constexpr auto thresh = 1e-4f;
    auto num_errors = 0;
    for (auto i = 0u; i < u_out_tensor.size(); ++i)
    {
      const auto& a = h_out_tensor.data()[i];
      const auto& b = u_out_tensor.data()[i];
      if (std::abs(a - b) > thresh)
      {
        std::cout << sara::format("[OUCH] i=%d me=%f trt=%f\n",  //
                                  i,                             //
                                  h_out_tensor.data()[i],        //
                                  u_out_tensor.data()[i]);
        ++num_errors;
      }
      if (num_errors > 20)
        break;
    }

    if (num_errors > 0)
    {
      std::cout << "h_out_tensor\n"
                << h_out_tensor[0][0].matrix().topLeftCorner(8, 8) << std::endl;
      std::cout << "u_out_tensor\n"
                << u_out_tensor[0].matrix().topLeftCorner(8, 8) << std::endl;

      SARA_CHECK(u_out_tensor.data());
    }

    BOOST_REQUIRE(std::equal(h_out_tensor.begin(), h_out_tensor.end(),
                             u_out_tensor.begin(),
                             [](const float& a, const float& b) {
                               return std::abs(a - b) < thresh;
                             }));
  }
}


#if defined(END_TO_END_YOLOV4)
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

  // Check the equality between the CPU implementation and the
  // TensorRT-based network.
  for (auto i = 0u; i < h_out_tensor.size(); ++i)
    BOOST_CHECK(std::equal(h_out_tensor[i].begin(), h_out_tensor[i].end(),
                           u_out_tensor[i].begin(),
                           [](const float& a, const float& b) {
                             return std::abs(a - b) < 1e-4f;
                           }));

  std::cout << "out 0 =\n" << u_out_tensor[0][0].matrix() << std::endl;
  std::cout << "out 1 =\n" << u_out_tensor[1][0].matrix() << std::endl;
}
#endif

BOOST_AUTO_TEST_SUITE_END()
