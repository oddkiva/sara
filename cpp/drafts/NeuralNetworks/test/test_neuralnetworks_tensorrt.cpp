#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <vector>


auto main() -> int
{
  namespace sara = DO::Sara;
  auto builder = sara::TensorRT::make_builder();

  // Instantiate a network and automatically manager its memory.
  auto network = sara::TensorRT::make_network(builder.get());


  // Instantiate an input data.
  auto image_tensor = network->addInput("image", nvinfer1::DataType::kFLOAT,
                                        nvinfer1::Dims3{1, 28, 28});


  // Create artificial weights.
  const auto conv1_kernel_weights_vector =
      std::vector<float>(5 * 5 * 1 * 20, 0.f);
  const auto conv1_bias_weights_vector = std::vector<float>(20, 0.f);


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
  const auto conv1_fn = network->addConvolution(
      *image_tensor,
      20,                    // number of filters
      {5, 5},                // kernel sizes
      conv1_kernel_weights,  // convolution kernel weights
      conv1_bias_weights);   // bias weight


  // Get the ouput tensor.
  /*  auto conv1 = */ conv1_fn->getOutput(0);

  return 0;
}
