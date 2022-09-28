#include <drafts/NeuralNetworks/TensorRT/IO.hpp>
#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>
#include <drafts/NeuralNetworks/TensorRT/YoloImpl.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <cassert>
#include <stdexcept>


namespace DO::Sara::TensorRT {

  nvinfer1::PluginFieldCollection YoloPluginCreator::_fc;
  std::vector<nvinfer1::PluginField> YoloPluginCreator::_plugin_attributes;


  auto YoloPlugin::getOutputDataType(
      [[maybe_unused]] const std::int32_t output_index,
      const nvinfer1::DataType* input_types,
      [[maybe_unused]] const std::int32_t num_inputs) const noexcept
      -> nvinfer1::DataType
  {
    return input_types[0];
  }

  auto YoloPlugin::isOutputBroadcastAcrossBatch(
      [[maybe_unused]] const std::int32_t output_index,
      [[maybe_unused]] const bool* input_is_broadcasted,
      [[maybe_unused]] const std::int32_t num_inputs) const noexcept -> bool
  {
    return false;
  }

  auto YoloPlugin::canBroadcastInputAcrossBatch(
      [[maybe_unused]] const std::int32_t input_index) const noexcept -> bool
  {
    return false;
  }

  auto YoloPlugin::clone() const noexcept -> nvinfer1::IPluginV2Ext*
  {
    try
    {
      auto plugin = new YoloPlugin{};
      plugin->setPluginNamespace(_namespace.c_str());
      SARA_DEBUG << "PLUGIN CREATED OK!" << std::endl;
      return plugin;
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << "EXCEPTION" << e.what() << std::endl;
    }

    return nullptr;
  }

  auto YoloPlugin::getPluginType() const noexcept -> const nvinfer1::AsciiChar*
  {
    return name;
  }

  auto YoloPlugin::getPluginVersion() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return version;
  }

  auto YoloPlugin::getNbOutputs() const noexcept -> std::int32_t
  {
    return 1;
  }

  auto YoloPlugin::getOutputDimensions(
      [[maybe_unused]] const std::int32_t index,
      [[maybe_unused]] const nvinfer1::Dims* inputs,
      [[maybe_unused]] const std::int32_t nb_input_dims) noexcept
      -> nvinfer1::Dims
  {
    return inputs[0];
  }

  auto YoloPlugin::initialize() noexcept -> std::int32_t
  {
    return 0;
  }

  auto YoloPlugin::terminate() noexcept -> void
  {
  }

  auto YoloPlugin::getWorkspaceSize(
      const std::int32_t /* max_batch_size */) const noexcept -> std::size_t
  {
    return 0;
  }

  //! TODO
  auto YoloPlugin::enqueue([[maybe_unused]] const std::int32_t batch_size,
                           void const* const* inputs, void* const* outputs,
                           [[maybe_unused]] void* workspace,
                           cudaStream_t stream) noexcept -> std::int32_t
  {
    try
    {
      SARA_DEBUG << "CALLING YOLO IMPLEMENTATION\n";
      const auto in = reinterpret_cast<const float*>(inputs[0]);
      const auto out = reinterpret_cast<float*>(outputs[0]);

      // The image is divided in a grid of size 13x13 or 26x26.
      // For each cell (i, j) of the grid, the YOLO layer predicts a fixed
      // number of boxes: typically 3.

      // The YOLO layer predicts a number of features for each of these box:
      // - four box coordinates: (x, y, w, h)
      static constexpr auto num_box_coordinates = 4;

      // Using the Bayes rule: Prob[class = i] = Prob[object] x Prob[class = i |
      // object],
      //
      // the YOLO layer estimates the following probabilities:
      //
      // - The probability that the box contains an object: 1
      // - The probability that the box contains an object of class i, if it
      //   does contain an object.
      //
      // There are 80 classes if YOLO is trained on COCO.
      const auto box_probabilities = 1 + _num_classes;
      const auto box_features = num_box_coordinates + box_probabilities;

      const auto c = _num_boxes_per_grid_cell * box_features;
      const auto size = c * _h * _w;

      yolo(in, out, size, c, _h, _w, _num_classes, _scale_x_y, stream);

      return 0;
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << e.what() << std::endl;
    }

    return -1;
  }

  auto YoloPlugin::getSerializationSize() const noexcept -> size_t
  {
    const auto yolo_parameter_byte_size =   //
        sizeof(_num_boxes_per_grid_cell) +  //
        sizeof(_num_classes) +              //
        sizeof(_h) +                        //
        sizeof(_w) +                        //
        sizeof(_scale_x_y);
    return yolo_parameter_byte_size;
  }

  auto YoloPlugin::serialize(void* buffer) const noexcept -> void
  {
    auto cbuf = reinterpret_cast<char*>(buffer);
    write_to_buffer(cbuf, _num_boxes_per_grid_cell);
    write_to_buffer(cbuf, _num_classes);
    write_to_buffer(cbuf, _h);
    write_to_buffer(cbuf, _w);
    write_to_buffer(cbuf, _scale_x_y);
  }

  auto YoloPlugin::destroy() noexcept -> void
  {
    delete this;
  }

  auto YoloPlugin::setPluginNamespace(
      const nvinfer1::AsciiChar* plugin_namespace) noexcept -> void
  {
    _namespace = plugin_namespace;
  }

  auto YoloPlugin::getPluginNamespace() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return _namespace.c_str();
  }

  //! TODO
  auto YoloPlugin::configurePlugin(
      [[maybe_unused]] const nvinfer1::PluginTensorDesc* inputs,
      [[maybe_unused]] const std::int32_t num_inputs,
      [[maybe_unused]] const nvinfer1::PluginTensorDesc* outputs,
      [[maybe_unused]] const std::int32_t num_outputs) noexcept -> void
  {
  }

  auto YoloPlugin::supportsFormatCombination(
      [[maybe_unused]] const std::int32_t pos,  //
      const nvinfer1::PluginTensorDesc* in_out,
      [[maybe_unused]] const std::int32_t nb_inputs,
      [[maybe_unused]] const std::int32_t nb_outputs) const noexcept -> bool
  {
    assert(nb_inputs == 1 || nb_outputs == 1 || pos == 0);

    return (in_out[0].type == nvinfer1::DataType::kHALF ||
            in_out[0].type == nvinfer1::DataType::kFLOAT) &&
           in_out[0].format == nvinfer1::PluginFormat::kLINEAR;
  }

  YoloPluginCreator::YoloPluginCreator()
  {
    _plugin_attributes.clear();
    _plugin_attributes.emplace_back("num_boxes_per_grid_cell", nullptr,
                                    nvinfer1::PluginFieldType::kINT32, 1);
    _plugin_attributes.emplace_back("num_classes", nullptr,
                                    nvinfer1::PluginFieldType::kINT32, 1);
    _plugin_attributes.emplace_back("height", nullptr,
                                    nvinfer1::PluginFieldType::kINT32, 1);
    _plugin_attributes.emplace_back("width", nullptr,
                                    nvinfer1::PluginFieldType::kINT32, 1);
    _plugin_attributes.emplace_back("scale_x_y", nullptr,
                                    nvinfer1::PluginFieldType::kFLOAT32, 1);

    _fc.fields = _plugin_attributes.data();
    _fc.nbFields = static_cast<std::int32_t>(_plugin_attributes.size());
  }

  auto YoloPluginCreator::getPluginName() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return YoloPlugin::name;
  }

  auto YoloPluginCreator::getPluginVersion() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return YoloPlugin::version;
  }

  auto YoloPluginCreator::getFieldNames() noexcept
      -> const nvinfer1::PluginFieldCollection*
  {
    return &_fc;
  }

  auto YoloPluginCreator::createPlugin(
      const nvinfer1::AsciiChar* trt_namespace,
      const nvinfer1::PluginFieldCollection* fc) noexcept
      -> nvinfer1::IPluginV2*
  {
    // All the necessary parameters for the YOLO layer.
    auto num_boxes_per_grid_cell = std::int32_t{};
    auto num_classes = std::int32_t{};
    auto h = std::int32_t{};
    auto w = std::int32_t{};
    auto scale_x_y = float{};

    // Parse the plugin field collection.
    const auto fields = fc->fields;
    const auto num_fields = fc->nbFields;
    for (auto i = 0; i < num_fields; ++i)
    {
      if (!std::strcmp(fields[i].name, "num_boxes_per_grid_cell"))
        num_boxes_per_grid_cell =
            *reinterpret_cast<const std::int32_t*>(fields[i].data);
      if (!std::strcmp(fields[i].name, "num_classes"))
        num_classes = *reinterpret_cast<const std::int32_t*>(fields[i].data);
      if (!std::strcmp(fields[i].name, "height"))
        h = *reinterpret_cast<const std::int32_t*>(fields[i].data);
      if (!std::strcmp(fields[i].name, "width"))
        w = *reinterpret_cast<const std::int32_t*>(fields[i].data);
      if (!std::strcmp(fields[i].name, "scale_x_y"))
        scale_x_y = *reinterpret_cast<const float*>(fields[i].data);
    }

    auto plugin = new YoloPlugin{
        num_boxes_per_grid_cell,  //
        num_classes,              //
        h, w,                     //
        scale_x_y                 //
    };
    plugin->setPluginNamespace(trt_namespace);
    return plugin;
  }

  auto YoloPluginCreator::getPluginNamespace() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return _namespace.c_str();
  }

  auto YoloPluginCreator::setPluginNamespace(
      [[maybe_unused]] const nvinfer1::AsciiChar* plugin_namespace) noexcept
      -> void
  {
    _namespace = plugin_namespace;
  }

  auto YoloPluginCreator::deserializePlugin(
      const nvinfer1::AsciiChar* plugin_namespace,  //
      const void* serial_data,
      [[maybe_unused]] const size_t serial_length) noexcept
      -> nvinfer1::IPluginV2*
  {
    try
    {
      auto buffer_ptr = reinterpret_cast<const char*>(serial_data);
      const auto num_boxes_per_grid_cell =
          read_from_buffer<std::int32_t>(buffer_ptr);
      const auto num_classes = read_from_buffer<std::int32_t>(buffer_ptr);
      const auto h = read_from_buffer<std::int32_t>(buffer_ptr);
      const auto w = read_from_buffer<std::int32_t>(buffer_ptr);
      const auto scale_x_y = read_from_buffer<float>(buffer_ptr);

      auto plugin = new YoloPlugin{num_boxes_per_grid_cell,  //
                                   num_classes,              //
                                   h, w,                     //
                                   scale_x_y};
      plugin->setPluginNamespace(plugin_namespace);
      return plugin;
    }
    catch (std::exception const& e)
    {
      SARA_DEBUG << "EXCEPTION: " << e.what() << std::endl;
    }
    return nullptr;
  }

}  // namespace DO::Sara::TensorRT
