#pragma once

#include <drafts/NeuralNetworks/Darknet/Network.hpp>

#include <NvInfer.h>


namespace DO::Sara::TensorRT {

  struct YoloV4TinyConverter
  {
    nvinfer1::INetworkDefinition* tnet;
    const std::vector<std::unique_ptr<Darknet::Layer>>& hnet;

    inline auto operator()() -> void
    {
      if (tnet == nullptr)
        throw std::runtime_error{"TensorRT network definition is NULL!"};
      if (hnet.empty())
        throw std::runtime_error{"Network is empty!"};

      const auto& input_layer = dynamic_cast<const Darknet::Input&>(*hnet[0]);
      auto input_tensor = tnet->addInput(
          "input",  //
          nvinfer1::DataType::kFLOAT,
          nvinfer1::Dims4{1, 3, input_layer.height(), input_layer.width()});

      for (auto i = 1u; i < hnet.size(); ++i)
      {
        
      }
    }
  };

}  // namespace DO::Sara::TensorRT
