#pragma once

#include <drafts/NeuralNetworks/Darknet/Network.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <NvInfer.h>


namespace DO::Sara::TensorRT {

  inline auto shape(const nvinfer1::ITensor& t)
      -> Eigen::Map<const Eigen::Vector4i>
  {
    const auto dims = t.getDimensions();
    return Eigen::Map<const Eigen::Vector4i>{dims.d, 4};
  }


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

    //! @brief zero-padding convolution.
    auto conv2d(nvinfer1::ITensor* x,  //
                const TensorView_<float, 4>& w, const Eigen::VectorXf& b,
                int stride) const -> nvinfer1::ITensor*
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
      const auto conv_fn = tnet->addConvolutionNd(  //
          *x,                                       // input tensor
          co,                                       // number of filters
          nvinfer1::DimsHW{kh, kw},                 // kernel sizes
          conv_weights,  // convolution kernel weights
          conv_bias);    // bias weights
      conv_fn->setStrideNd(nvinfer1::DimsHW{stride, stride});
      conv_fn->setPaddingNd(nvinfer1::DimsHW{kh / 2, kw / 2});
      const auto y = conv_fn->getOutput(0);
      return y;
    }

    auto operator()() -> void
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

      // The list of intermediate feature maps.
      auto fmaps = std::vector<nvinfer1::ITensor*>{};
      fmaps.push_back(input_tensor);

      SARA_DEBUG << "Shape 0 : " << shape(*fmaps.back()).transpose()
                 << std::endl;

      for (auto layer_idx = 1u; layer_idx < hnet.size(); ++layer_idx)
      {
        // Update the input.
        const auto& layer_type = hnet[layer_idx]->type;
        if (layer_type == "convolutional")
        {
          SARA_DEBUG << "Converting convolutional layer " << layer_idx
                     << " to TRT" << std::endl;
          const auto& conv_layer =
              dynamic_cast<const Darknet::Convolution&>(*hnet[layer_idx]);
          std::cout << conv_layer << std::endl;

          // It's always the last one in Darknet cfg file.
          auto& x = fmaps.back();
          auto y = conv2d(x, conv_layer.weights.w, conv_layer.weights.b,
                          conv_layer.stride);
          fmaps.push_back(y);

          SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                     << shape(*fmaps.back()).transpose() << std::endl;
        }
        else if (layer_type == "route")
        {
          const auto& route_layer =
              dynamic_cast<const Darknet::Route&>(*hnet[layer_idx]);

          if (route_layer.layers.size() == 1)
          {
            SARA_DEBUG << "convert route-slice layer " << layer_idx << "("
                       << hnet[layer_idx]->type << ")" << std::endl;
            std::cout << route_layer << std::endl;

            // Retrieve the index of the input tensor.
            const auto& rel_idx = route_layer.layers.front();
            const auto glob_idx =
                rel_idx < 0 ? layer_idx + rel_idx
                            : rel_idx + 1 /* because of the input layer. */;

            // Only keep the last half channels in the feature maps.
            auto& x = fmaps[glob_idx];
            const auto x_dims = x->getDimensions();
            const auto n = x_dims.d[0];
            const auto c_start =
                x_dims.d[1] * route_layer.group_id / route_layer.groups;
            const auto c_size = x_dims.d[1] / route_layer.groups;
            const auto h = x_dims.d[2];
            const auto w = x_dims.d[3];
            const auto start = nvinfer1::Dims4{n, c_start, h, w};
            const auto size = nvinfer1::Dims4{n, c_size, h, w};
            const auto stride = nvinfer1::Dims4{1, 1, 1, 1};

            const auto slice_layer = tnet->addSlice(*x, start, size, stride);

            const auto y = slice_layer->getOutput(0);
            fmaps.push_back(y);

            SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                       << shape(*fmaps.back()).transpose() << std::endl;
            SARA_DEBUG << "TRT start : "
                       << Eigen::Map<const Eigen::RowVector4i>(start.d)
                       << std::endl;
            SARA_DEBUG << "TRT size : "
                       << Eigen::Map<const Eigen::RowVector4i>(size.d)
                       << std::endl;
            SARA_DEBUG << "TRT stride : "
                       << Eigen::Map<const Eigen::RowVector4i>(stride.d)
                       << std::endl;
          }
          else
          {
            SARA_DEBUG << "convert route-concat layer " << layer_idx << "("
                       << hnet[layer_idx]->type << ")" << std::endl;
            std::cout << route_layer << std::endl;

            auto xs = std::vector<nvinfer1::ITensor*>{};
            for (const auto& rel_idx : route_layer.layers)
            {
              // Retrieve the index of the input tensor.
              const auto glob_idx =
                  rel_idx < 0 ? layer_idx + rel_idx
                              : rel_idx + 1 /* because of the input layer. */;
              xs.push_back(fmaps[glob_idx]);
            }

            const auto concat_layer =
                tnet->addConcatenation(xs.data(), xs.size());
            const auto y = concat_layer->getOutput(0);
            fmaps.push_back(y);
            SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                       << shape(*fmaps.back()).transpose() << std::endl;
          }
        }
        else if (layer_type == "maxpool")
        {
          const auto& maxpool_layer =
              dynamic_cast<const Darknet::MaxPool&>(*hnet[layer_idx]);
          SARA_DEBUG << "convert maxpool layer " << layer_idx << "("
                     << hnet[layer_idx]->type << ")" << std::endl;
          std::cout << maxpool_layer << std::endl;

          const auto size = maxpool_layer.size;
          const auto stride = maxpool_layer.stride;

          const auto x = fmaps.back();
          auto trt_maxpool_layer = tnet->addPoolingNd(
              *x, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
          trt_maxpool_layer->setStrideNd(nvinfer1::DimsHW{stride, stride});

          auto y = trt_maxpool_layer->getOutput(0);
          fmaps.push_back(y);

          SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                     << shape(*fmaps.back()).transpose() << std::endl;
        }
        else
        {
          SARA_DEBUG << "TODO: convert layer " << layer_idx << "("
                     << hnet[layer_idx]->type << ")" << std::endl;
          std::cout << *hnet[layer_idx] << std::endl;
          break;
        }
      }

      tnet->markOutput(*fmaps.back());
    }
  };

}  // namespace DO::Sara::TensorRT
