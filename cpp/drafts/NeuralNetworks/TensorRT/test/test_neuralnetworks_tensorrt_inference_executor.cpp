#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/InferenceExecutor"

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>

#include <drafts/NeuralNetworks/TensorRT/DarknetParser.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>
#include <drafts/NeuralNetworks/TensorRT/InferenceExecutor.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace trt = sara::TensorRT;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_inference_executor)
{
  // Build the network model.
  auto net_builder = trt::make_builder();
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";
  auto net = trt::make_yolo_v4_tiny_network(net_builder,  //
                                            yolov4_tiny_dirpath.string());

  // Serialized the network.
  auto serialized_net = trt::serialize_network_into_plan(net_builder, net,  //
                                                         /* use_fp16 */ false);

  auto inference_executor = trt::InferenceExecutor{serialized_net};

  // Prepare the input tensor
  const auto image = sara::imread<sara::Rgb8>(src_path("data/dog.jpg"));

  // Resize the image to the network input sizes.
  const auto image_resized =
      sara::resize(image, {416, 416})
          .convert<sara::Rgb32f>();
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});

  // Resize the host tensor.
  auto u_in_tensor = trt::InferenceExecutor::PinnedTensor<float, 3>{};
  u_in_tensor.resize(3, image_resized.height(), image_resized.width());
  std::copy(image_tensor.begin(), image_tensor.end(), u_in_tensor.begin());

}

BOOST_AUTO_TEST_SUITE_END()
