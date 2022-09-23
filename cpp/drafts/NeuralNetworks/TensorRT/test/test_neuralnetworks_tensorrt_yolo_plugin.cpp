#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT"

#include <boost/test/unit_test.hpp>

#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>


namespace sara = DO::Sara;
namespace trt = sara::TensorRT;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_yolo_plugin)
{
  auto creator = nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSAFE_GPU);
  auto yoloPlugin = creator->getPluginCreator("yolo", "0.1");
}

BOOST_AUTO_TEST_SUITE_END()
