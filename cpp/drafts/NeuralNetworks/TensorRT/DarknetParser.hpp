#pragma once

#include <drafts/NeuralNetworks/Darknet/Network.hpp>
#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <NvInfer.h>

#include <stdexcept>


namespace DO::Sara::TensorRT {

  inline auto shape(const nvinfer1::ITensor& t) -> Eigen::Vector4i
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

    auto make_input_rgb_tensor(const int w, const int h) const
        -> nvinfer1::ITensor*
    {
      return tnet->addInput("input",  //
                            nvinfer1::DataType::kFLOAT,
                            nvinfer1::Dims4{1, 3, h, w});
    }

    //! @brief zero-padding convolution.
    auto conv2d(nvinfer1::ITensor* x,  //
                const TensorView_<float, 4>& w, const Eigen::VectorXf& b,
                const int stride, const std::string& activation_layer,
                const std::optional<std::string>& name = std::nullopt) const
        -> nvinfer1::ITensor*
    {
      // Encapsulate the weights using TensorRT data structures.
      const auto conv_weights = nvinfer1::Weights{
          nvinfer1::DataType::kFLOAT, reinterpret_cast<const void*>(w.data()),
          static_cast<std::int64_t>(w.size())};
      const auto conv_bias = nvinfer1::Weights{
          nvinfer1::DataType::kFLOAT, reinterpret_cast<const void*>(b.data()),
          static_cast<std::int64_t>(b.size())};

      const auto co = w.size(0);
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

      // Set a default for debugging.
      using namespace std::string_literals;
      auto conv_layer_name = "fused_conv_bn"s;
      if (name.has_value())
        conv_layer_name = *name + "/" + conv_layer_name;
      conv_fn->setName(conv_layer_name.c_str());

      // Get the convolution output.
      auto y = conv_fn->getOutput(0);

      // Apply the activation.
      if (activation_layer == "leaky")
      {
        const auto leaky_fn =
            tnet->addActivation(*y, nvinfer1::ActivationType::kLEAKY_RELU);
        leaky_fn->setAlpha(0.1f);

        auto leaky_layer_name = "leaky"s;
        if (name.has_value())
          leaky_layer_name = *name + "/" + leaky_layer_name;

        leaky_fn->setName(leaky_layer_name.c_str());
        y = leaky_fn->getOutput(0);
      }
      else if (activation_layer == "linear")
      {
      }
      else
        throw std::invalid_argument{"activation layer: " + activation_layer +
                                    " is not implemented!"};


      // The output.
      return y;
    }

    auto add_conv2d_layer(const int layer_idx,
                          std::vector<nvinfer1::ITensor*>& fmaps) const -> void
    {
      SARA_DEBUG << "Converting convolutional layer " << layer_idx << " to TRT"
                 << std::endl;
      const auto& conv_layer =
          dynamic_cast<const Darknet::Convolution&>(*hnet[layer_idx]);
      std::cout << conv_layer << std::endl;

      // It's always the last one in Darknet cfg file.
      auto& x = fmaps.back();
      auto y = conv2d(x, conv_layer.weights.w, conv_layer.weights.b,
                      conv_layer.stride, conv_layer.activation,
                      "conv_bn_" + conv_layer.activation + "_" +
                          std::to_string(layer_idx));
      fmaps.push_back(y);

      SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                 << shape(*fmaps.back()).transpose() << std::endl;
    }

    auto add_slice_layer(const int layer_idx,
                         std::vector<nvinfer1::ITensor*>& fmaps) const -> void
    {
      const auto& route_layer =
          dynamic_cast<const Darknet::Route&>(*hnet[layer_idx]);
      SARA_DEBUG << "convert route-slice layer " << layer_idx << "("
                 << route_layer.type << ")" << std::endl;
      std::cout << route_layer << std::endl;

      // Retrieve the index of the input tensor.
      const auto& rel_idx = route_layer.layers.front();
      const auto glob_idx = rel_idx < 0
                                ? layer_idx + rel_idx
                                : rel_idx + 1 /* because of the input layer. */;

      // Only keep the last half channels in the feature maps.
      auto& x = fmaps[glob_idx];
      const auto x_dims = x->getDimensions();
      const auto c_start =
          route_layer.group_id != -1
              ? x_dims.d[1] * route_layer.group_id / route_layer.groups
              : 0;
      const auto c_size = x_dims.d[1] / route_layer.groups;
      const auto h = x_dims.d[2];
      const auto w = x_dims.d[3];
      const auto start = nvinfer1::Dims4{0, c_start, 0, 0};
      const auto size = nvinfer1::Dims4{1, c_size, h, w};
      const auto stride = nvinfer1::Dims4{1, 1, 1, 1};

      const auto trt_slice_layer = tnet->addSlice(*x, start, size, stride);
      trt_slice_layer->setName(("slice_" + std::to_string(layer_idx)).c_str());

      const auto y = trt_slice_layer->getOutput(0);
      fmaps.push_back(y);

      SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                 << shape(*fmaps.back()).transpose() << std::endl;
      SARA_DEBUG << "TRT start : "
                 << Eigen::Map<const Eigen::RowVector4i>(start.d) << std::endl;
      SARA_DEBUG << "TRT size : "
                 << Eigen::Map<const Eigen::RowVector4i>(size.d) << std::endl;
      SARA_DEBUG << "TRT stride : "
                 << Eigen::Map<const Eigen::RowVector4i>(stride.d) << std::endl;
    }

    auto add_concat_layer(const int layer_idx,
                          std::vector<nvinfer1::ITensor*>& fmaps) const -> void
    {
      const auto& route_layer =
          dynamic_cast<const Darknet::Route&>(*hnet[layer_idx]);
      SARA_DEBUG << "convert route-concat layer " << layer_idx << "("
                 << route_layer.type << ")" << std::endl;
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

      const auto trt_concat_layer =
          tnet->addConcatenation(xs.data(), xs.size());
      trt_concat_layer->setName(
          ("concat_" + std::to_string(layer_idx)).c_str());

      const auto y = trt_concat_layer->getOutput(0);
      fmaps.push_back(y);
      SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                 << shape(*fmaps.back()).transpose() << std::endl;
    }

    auto add_maxpool_layer(const int layer_idx,
                           std::vector<nvinfer1::ITensor*>& fmaps) const -> void
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

      trt_maxpool_layer->setName(
          ("maxpool_" + std::to_string(layer_idx)).c_str());

      auto y = trt_maxpool_layer->getOutput(0);
      fmaps.push_back(y);

      SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                 << shape(*fmaps.back()).transpose() << std::endl;
    }

    auto add_upsample_layer(const int layer_idx,
                            std::vector<nvinfer1::ITensor*>& fmaps) const
        -> void
    {
      const auto& upsample_layer =
          dynamic_cast<const Darknet::Upsample&>(*hnet[layer_idx]);
      SARA_DEBUG << "convert layer " << layer_idx << "(" << upsample_layer.type
                 << ")" << std::endl;
      std::cout << upsample_layer << std::endl;

      const auto x = fmaps.back();

      // Define the TensorRT upsample layer.
      const auto trt_upsample_layer = tnet->addResize(*x);
      // N.B.: it really is the nearest interpolation mode.
      trt_upsample_layer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
      const auto out_dims = nvinfer1::Dims4{
          upsample_layer.output_sizes(0),
          upsample_layer.output_sizes(1),
          upsample_layer.output_sizes(2),
          upsample_layer.output_sizes(3),
      };
      trt_upsample_layer->setOutputDimensions(out_dims);

      const auto y = trt_upsample_layer->getOutput(0);
      fmaps.push_back(y);

      SARA_DEBUG << "TRT Shape " << layer_idx << " : "
                 << shape(*fmaps.back()).transpose() << std::endl;
    }

    auto add_yolo_layer(const int layer_idx,
                        std::vector<nvinfer1::ITensor*>& fmaps) const -> void
    {
      const auto& yolo_layer =
          dynamic_cast<const Darknet::Yolo&>(*hnet[layer_idx]);
      SARA_DEBUG << "convert yolo layer " << layer_idx << "("
                 << hnet[layer_idx]->type << ")" << std::endl;
      std::cout << yolo_layer << std::endl;

      const auto plugin_registry = getPluginRegistry();
      assert(plugin_registry != nullptr);
      const auto yolo_plugin_creator = plugin_registry->getPluginCreator(
          YoloPlugin::name, YoloPlugin::version);
      assert(yolo_plugin_creator != nullptr);

      static constexpr auto delete_plugin =
          [](nvinfer1::IPluginV2* const plugin) { plugin->destroy(); };
      SARA_DEBUG << "Creating TensorRT-YOLO plugin...\n";

      // Create the plugin field collection.
      auto fields = std::vector<nvinfer1::PluginField>{};
      const auto num_boxes_per_grid_cell =
          static_cast<std::int32_t>(yolo_layer.mask.size());
      fields.emplace_back("num_boxes_per_grid_cell", &num_boxes_per_grid_cell,
                          nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back("num_classes", &yolo_layer.classes,
                          nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back("height", &yolo_layer.output_sizes(2),
                          nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back("width", &yolo_layer.output_sizes(3),
                          nvinfer1::PluginFieldType::kINT32, 1);
      fields.emplace_back("scale_x_y", &yolo_layer.scale_x_y,
                          nvinfer1::PluginFieldType::kFLOAT32, 1);
      SARA_CHECK(num_boxes_per_grid_cell);
      SARA_CHECK(yolo_layer.classes);

      auto fc = nvinfer1::PluginFieldCollection{};
      fc.fields = fields.data();
      fc.nbFields = static_cast<std::int32_t>(fields.size());

      // Create the YOLO plugin.
      const auto yolo_plugin =
          std::unique_ptr<nvinfer1::IPluginV2, decltype(delete_plugin)>{
              yolo_plugin_creator->createPlugin("", &fc), delete_plugin};
      assert(yolo_plugin.get() != nullptr);

      auto x = fmaps.back();
      auto trt_yolo_layer = tnet->addPluginV2(&x, 1, *yolo_plugin);
      auto y = trt_yolo_layer->getOutput(0);
      fmaps.push_back(y);
    }

    auto operator()(const std::size_t max_layers =
                        std::numeric_limits<std::size_t>::max()) -> void
    {
      if (tnet == nullptr)
        throw std::runtime_error{"TensorRT network definition is NULL!"};
      if (hnet.empty())
        throw std::runtime_error{"Network is empty!"};

      SARA_DEBUG << "Creating the network from scratch!" << std::endl;

      // Define the input tensor.
      const auto& input_layer = dynamic_cast<const Darknet::Input&>(*hnet[0]);
      auto input_tensor = make_input_rgb_tensor(input_layer.width(),  //
                                                input_layer.height());

      // The list of intermediate feature maps.
      auto fmaps = std::vector<nvinfer1::ITensor*>{};
      fmaps.push_back(input_tensor);
      SARA_DEBUG << "Shape 0 : " << shape(*fmaps.back()).transpose()
                 << std::endl;

      for (auto layer_idx = 1u; layer_idx < hnet.size(); ++layer_idx)
      {
        if (layer_idx > max_layers)
          break;

        // Update the input.
        const auto& layer_type = hnet[layer_idx]->type;
        if (layer_type == "convolutional")
          add_conv2d_layer(layer_idx, fmaps);
        else if (layer_type == "route")
        {
          const auto& route_layer =
              dynamic_cast<const Darknet::Route&>(*hnet[layer_idx]);

          if (route_layer.layers.size() == 1)
            add_slice_layer(layer_idx, fmaps);
          else
            add_concat_layer(layer_idx, fmaps);
        }
        else if (layer_type == "maxpool")
          add_maxpool_layer(layer_idx, fmaps);
        else if (layer_type == "upsample")
          add_upsample_layer(layer_idx, fmaps);
        else if (layer_type == "yolo")
          add_yolo_layer(layer_idx, fmaps);
        else
        {
          SARA_DEBUG << "TODO: convert layer " << layer_idx << "("
                     << hnet[layer_idx]->type << ")" << std::endl;
          std::cout << *hnet[layer_idx] << std::endl;
          throw std::runtime_error{"TENSORRT LAYER CONVERSION " + layer_type +
                                   " NOT IMPLEMENTED!"};
        }
      }

      tnet->markOutput(*fmaps.back());
    }
  };

}  // namespace DO::Sara::TensorRT
