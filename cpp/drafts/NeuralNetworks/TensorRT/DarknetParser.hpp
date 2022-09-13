#pragma once

#include <drafts/NeuralNetworks/Darknet/Network.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <NvInfer.h>


namespace DO::Sara::TensorRT {

  struct YoloV4TinyConverter
  {
    using TrtNet = nvinfer1::INetworkDefinition;
    using HostNet = std::vector<std::unique_ptr<Darknet::Layer>>;

    TrtNet* tnet;
    const HostNet& hnet;

    YoloV4TinyConverter(TrtNet* tnet, const HostNet& hnet)
      : tnet{tnet}
      , hnet{hnet}
    {
    }

    inline auto conv2d(nvinfer1::ITensor* x,  //
                       const TensorView_<float, 4>& w,
                       const Eigen::VectorXf& b) const -> nvinfer1::ITensor*
    {
      // Encapsulate the weights using TensorRT data structures.
      const auto conv_weights = nvinfer1::Weights{
          nvinfer1::DataType::kFLOAT, reinterpret_cast<const void*>(w.data()),
          static_cast<std::int64_t>(w.size())};
      const auto conv_bias = nvinfer1::Weights{
          nvinfer1::DataType::kFLOAT, reinterpret_cast<const void*>(b.data()),
          static_cast<std::int64_t>(b.size())};

      const auto co = w.size(0);
      // const auto ci = w.size(1);
      const auto kh = w.size(2);
      const auto kw = w.size(3);

      // Create a convolutional function.
      const auto conv_fn =
          tnet->addConvolutionNd(*x,
                                 co,                        // number of filters
                                 nvinfer1::DimsHW{kh, kw},  // kernel sizes
                                 conv_weights,  // convolution kernel weights
                                 conv_bias);    // bias weights
                                                //
      const auto y = conv_fn->getOutput(0);
      return y;
    }

    inline auto operator()() -> void
    {
      if (tnet == nullptr)
        throw std::runtime_error{"TensorRT network definition is NULL!"};
      if (hnet.empty())
        throw std::runtime_error{"Network is empty!"};

      SARA_DEBUG << termcolor::green << "Creating the network from scratch!"
                 << std::endl;

      const auto& input_layer = dynamic_cast<const Darknet::Input&>(*hnet[0]);
      auto input_tensor = tnet->addInput(
          "input",  //
          nvinfer1::DataType::kFLOAT,
          nvinfer1::Dims4{1, 3, input_layer.height(), input_layer.width()});

      auto x = input_tensor;
      auto y = x;

      for (auto i = 1u; i < hnet.size(); ++i)
      {
        const auto& layer_type = hnet[i]->type;
        if (layer_type == "convolutional")
        {
          SARA_DEBUG << "Converting convolutional layer " << i << " to TRT" << std::endl;
          const auto& conv_layer =
              dynamic_cast<const Darknet::Convolution&>(*hnet[i]);
          y = conv2d(x, conv_layer.weights.w, conv_layer.weights.b);
        }
        else if (layer_type == "route")
        {
        }

        x = y;
      }
    }
  };

}  // namespace DO::Sara::TensorRT
