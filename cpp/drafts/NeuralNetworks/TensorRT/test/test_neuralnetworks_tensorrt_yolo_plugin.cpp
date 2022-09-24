#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT"

#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;
namespace trt = sara::TensorRT;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_yolo_plugin)
{
  auto plugin_registry =
      nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kDEFAULT);
  BOOST_CHECK_NE(plugin_registry, nullptr);

  auto yolo_plugin_creator = trt::YoloPluginCreator{};
  // plugin_registry->registerCreator(yolo_plugin_creator, "TensorRT/Tiny-Yolo-V4");

  // auto yolo_plugin = ;
  // BOOST_CHECK_NE(yolo_plugin, nullptr);
  // SARA_CHECK(yolo_plugin->getPluginName());
}

BOOST_AUTO_TEST_SUITE_END()
