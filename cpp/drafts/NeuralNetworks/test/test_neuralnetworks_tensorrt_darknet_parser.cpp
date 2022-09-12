#define BOOST_TEST_MODULE "NeuralNetworks/TensorRT"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Shakti/Cuda/MultiArray.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <termcolor/termcolor.hpp>

#include <fstream>
#include <sstream>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace trt = sara::TensorRT;


template <typename T, int N>
using PinnedTensor = sara::Tensor_<float, N, shakti::PinnedMemoryAllocator>;


BOOST_AUTO_TEST_SUITE(TestTensorRT)

BOOST_AUTO_TEST_CASE(test_network_build_process)
{
  auto builder = sara::TensorRT::make_builder();

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
}

BOOST_AUTO_TEST_SUITE_END()
