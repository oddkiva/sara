#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT/InferenceExecutor"

#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>

#include <drafts/NeuralNetworks/Darknet/Parser.hpp>
#include <drafts/NeuralNetworks/TensorRT/DarknetParser.hpp>
#include <drafts/NeuralNetworks/TensorRT/IO.hpp>
#include <drafts/NeuralNetworks/TensorRT/InferenceExecutor.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace d = sara::Darknet;
namespace trt = sara::TensorRT;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_inference_executor)
{
  // Load the network on the host device (CPU).
  const auto data_dir_path = fs::canonical(fs::path{src_path("data")});
  const auto yolov4_tiny_dirpath = data_dir_path / "trained_models";

  // Convert it into a TensorRT network object.
  auto serialized_net = trt::convert_yolo_v4_tiny_network_from_darknet(
      yolov4_tiny_dirpath.string());
  auto inference_executor = trt::InferenceExecutor{serialized_net};

  // Prepare the input tensor
  const auto image = sara::imread<sara::Rgb8>(src_path("data/dog.jpg"));

  // Resize the image to the network input sizes.
  const auto image_resized =
      sara::resize(image, {416, 416}).convert<sara::Rgb32f>();
  const auto image_tensor =
      sara::tensor_view(image_resized)
          .reshape(Eigen::Vector4i{1, image_resized.height(),
                                   image_resized.width(), 3})
          .transpose({0, 3, 1, 2});

  // Resize the host tensor.
  auto cuda_in_tensor = trt::InferenceExecutor::PinnedTensor<float, 3>{
      3, image_resized.height(), image_resized.width()};
  std::copy(image_tensor.begin(), image_tensor.end(), cuda_in_tensor.begin());

  auto cuda_out_tensor = std::array{
      trt::InferenceExecutor::PinnedTensor<float, 3>{255, 13, 13},
      trt::InferenceExecutor::PinnedTensor<float, 3>{255, 26, 26}  //
  };

  inference_executor(cuda_in_tensor, cuda_out_tensor, true);

  std::cout << "out 0 =\n" << cuda_out_tensor[0][0].matrix() << std::endl;
  std::cout << "out 1 =\n" << cuda_out_tensor[1][0].matrix() << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
