#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/YoloX-Tiny-ONNX-Conversion"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>

#include <filesystem>


namespace fs = std::filesystem;

namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace trt = sara::TensorRT;

namespace nvonnx = nvonnxparser;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<T, N, shakti::PinnedMemoryAllocator>;


static inline auto shape(const nvinfer1::ITensor& t) -> Eigen::Vector4i
{
  const auto dims = t.getDimensions();
  return Eigen::Map<const Eigen::Vector4i>{dims.d, 4};
}

static constexpr auto Kb = 1ul << 10;
static constexpr auto Mb = 1ul << 20;
static constexpr auto Gb = 1ul << 30;

auto serialize_onnx_model_as_tensort_engine(
    const fs::path& onnx_filepath,
    const std::size_t gpu_memory_budget = 6ul * Gb) -> trt::HostMemoryUniquePtr
{
  // Instantiate a network and automatically manage its memory.
  auto builder = trt::make_builder();
  auto network = trt::make_network(builder.get());

  // Instantiate an ONNX parser and read the ONNX model file.
  auto onnx_parser = trt::OnnxParserUniquePtr{
      nvonnxparser::createParser(*network, trt::Logger::instance()),
      &trt::onnx_parser_deleter  //
  };

  const auto parsed_successfully = onnx_parser->parseFromFile(
      yolox_tiny_onnx_filepath.string().c_str(),
      static_cast<std::int32_t>(nvinfer1::ILogger::Severity::kWARNING));
  for (auto i = 0; i < onnx_parser->getNbErrors(); ++i)
    std::cerr << "[ONNX parse error] " << onnx_parser->getError(i)->desc()
              << std::endl;
  if (parsed_successfully)
    throw std::runtime_error{"Failed to parse the ONNX model successfully!"};

  // Prepare the model optimization with a GPU memory budget
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),  //
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             gpu_memory_budget);

  // Optimize and serialize the network definition and weights for TensorRT.
  auto plan = trt::HostMemoryUniquePtr{
      builder->buildSerializedNetwork(*network, *config),  //
      trt::host_memory_deleter};
  if (plan.get() == nullptr)
    throw std::runtime_error{"Failed to serialize the ONNX model!"};
  if (plan->size() == 0)
    throw std::runtime_error{"The byte size of the serialized engine is 0!"};
}


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_yolox_tiny_onnx_conversion_to_trt_serialized_engine)
{
  // Instantiate a network and automatically manage its memory.
  auto builder = trt::make_builder();
  auto network = trt::make_network(builder.get());

  // Make sure we downloaded the YOLO-X Tiny ONNX model provided by Megvii:
  //
  // It is available here:
  // https://yolox.readthedocs.io/en/latest/demo/onnx_readme.html#download-onnx-models
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolox_tiny_onnx_filepath =
      data_dir_path / "trained_models" / "yolox_tiny.onnx";
  BOOST_CHECK(fs::exists(yolox_tiny_onnx_filepath));

  // Instantiate an ONNX parser and read the ONNX model file.
  auto onnx_parser = trt::OnnxParserUniquePtr{
      nvonnxparser::createParser(*network, trt::Logger::instance()),
      &trt::onnx_parser_deleter  //
  };

  // The file parsing should be successful.
  const auto parsed_successfully = onnx_parser->parseFromFile(
      yolox_tiny_onnx_filepath.string().c_str(),
      static_cast<std::int32_t>(nvinfer1::ILogger::Severity::kWARNING));
  BOOST_CHECK(parsed_successfully);
  for (auto i = 0; i < onnx_parser->getNbErrors(); ++i)
    std::cout << onnx_parser->getError(i)->desc() << std::endl;
  BOOST_CHECK_EQUAL(onnx_parser->getNbErrors(), 0);

  // There is only one input tensor.
  BOOST_CHECK_EQUAL(network->getNbInputs(), 1);
  SARA_CHECK(network->getInput(0)->getName());
  SARA_CHECK(shape(*network->getInput(0)).transpose());

  // There is only one output tensor.
  BOOST_CHECK_EQUAL(network->getNbOutputs(), 1);
  SARA_CHECK(network->getOutput(0)->getName());
  SARA_CHECK(shape(*network->getOutput(0)).transpose());

  // Prepare the model optimization with a GPU memory budget
  static constexpr auto Gb = 1ul << 30;
  static constexpr auto gpu_memory_budget = 6ul * Gb;
  auto config = trt::ConfigUniquePtr{builder->createBuilderConfig(),  //
                                     &trt::config_deleter};
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             gpu_memory_budget);

  // Optimize and serialize the network definition and weights for TensorRT.
  auto plan = trt::HostMemoryUniquePtr{
      builder->buildSerializedNetwork(*network, *config),  //
      trt::host_memory_deleter};
  BOOST_CHECK_NE(plan.get(), nullptr);
  BOOST_CHECK_NE(plan->size(), 0);
  SARA_CHECK(plan->size());

  // We can save the model.
}

BOOST_AUTO_TEST_SUITE_END()
