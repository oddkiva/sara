#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT"

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>
#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Cuda/MultiArray/PinnedMemoryAllocator.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace trt = sara::TensorRT;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<float, N, shakti::PinnedMemoryAllocator>;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_that_the_yolo_plugin_is_automatically_registered)
{
  auto plugin_registry = getPluginRegistry();
  BOOST_CHECK_NE(plugin_registry, nullptr);

  auto yolo_plugin_creator = plugin_registry->getPluginCreator(
      trt::YoloPlugin::name, trt::YoloPlugin::version);
  BOOST_CHECK_NE(yolo_plugin_creator, nullptr);
  BOOST_CHECK_NE(std::string{yolo_plugin_creator->getPluginName()}.find(
                     trt::YoloPlugin::name),
                 std::string::npos);
  BOOST_CHECK_NE(std::string{yolo_plugin_creator->getPluginVersion()}.find(
                     trt::YoloPlugin::version),
                 std::string::npos);

  auto num_plugin_creators = std::int32_t{0};
  const auto plugin_creators =
      plugin_registry->getPluginCreatorList(&num_plugin_creators);
  BOOST_CHECK_NE(plugin_creators, nullptr);

  const auto plugin_creator_first = plugin_creators;
  const auto plugin_creator_last = plugin_creators + num_plugin_creators;
  const auto yolo_creator_it = std::find_if(
      plugin_creator_first, plugin_creator_last,
      [yolo_plugin_creator](const auto& plugin_creator) {
        const auto plugin_name = std::string{plugin_creator->getPluginName()};
        return plugin_name.find(yolo_plugin_creator->getPluginName()) !=
               std::string::npos;
      });
  BOOST_CHECK_NE(yolo_creator_it, plugin_creator_last);

  auto yolo_plugin = yolo_plugin_creator->createPlugin("yolo", nullptr);
  SARA_CHECK(yolo_plugin->getPluginNamespace());
  SARA_CHECK(yolo_plugin->getPluginType());

  // Resize the host tensor.
  auto u_in_tensor = PinnedTensor<float, 3>{};
  // u_in_tensor.resize(3, input_layer.height(), input_layer.width());

  // Inspect the TensorRT log output: there is no padding!
  auto u_out_tensor = PinnedTensor<float, 3>{};
  // u_out_tensor.resize{out_sizes(1), out_sizes(2), out_sizes(3)};

  const auto batch_size = 0;
  auto inputs = std::vector<const void *>{};
  auto outputs = std::vector<void *>{};
  const auto workspace = nullptr;

  auto cuda_stream = trt::make_cuda_stream();
  BOOST_CHECK_EQUAL(yolo_plugin->enqueue(batch_size, inputs.data(),
                                         outputs.data(), workspace,
                                         *cuda_stream),
                    0);
}

BOOST_AUTO_TEST_SUITE_END()
