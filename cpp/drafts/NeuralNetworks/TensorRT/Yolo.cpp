#include <drafts/NeuralNetworks/TensorRT/Yolo.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <cassert>
#include <stdexcept>


namespace DO::Sara::TensorRT {

  nvinfer1::PluginFieldCollection YoloPluginCreator::_fc;


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
      return plugin;
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << e.what() << std::endl;
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

  //! TODO
  auto YoloPlugin::initialize() noexcept -> std::int32_t
  {
    return 1;
  }

  //! TODO
  auto YoloPlugin::terminate() noexcept -> void
  {
  }

  //! TODO
  auto YoloPlugin::getWorkspaceSize(
      const std::int32_t /* maxBatchSize */) const noexcept -> std::size_t
  {
    return 1;
  }

  //! TODO
  auto YoloPlugin::enqueue([[maybe_unused]] const std::int32_t batchSize,
                           [[maybe_unused]] void const* const* inputs,
                           [[maybe_unused]] void* const* outputs,
                           [[maybe_unused]] void* workspace,
                           [[maybe_unused]] cudaStream_t stream) noexcept
      -> std::int32_t
  {
    try
    {
#ifdef DO_SOMETHING_WITH_CUDA
      char* output = reinterpret_cast<char*>(outputs[0]);
      // expand to batch size
      for (int i = 0; i < batchSize; i++)
      {
        auto ret = cudaMemcpyAsync(output + i * mCopySize, inputs[1], mCopySize,
                                   cudaMemcpyDeviceToDevice, stream);
        if (ret != cudaSuccess)
          return ret;
      }
      return 0;
#else
#endif
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << e.what() << std::endl;
    }

    return -1;
  }

  //! TODO
  auto YoloPlugin::getSerializationSize() const noexcept -> size_t
  {
    return 0;
  }

  //! TODO
  auto YoloPlugin::serialize(void* /* buffer */) const noexcept -> void
  {
  }

  //! TODO
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

  //! TODO
  auto YoloPlugin::supportsFormatCombination(
      [[maybe_unused]] const std::int32_t pos,  //
      const nvinfer1::PluginTensorDesc* in_out,
      [[maybe_unused]] const std::int32_t nb_inputs,
      [[maybe_unused]] const std::int32_t nb_outputs) const noexcept -> bool
  {
    assert(nb_inputs == 1 || nb_outputs == 1 || pos == 0);

    return in_out[0].type == nvinfer1::DataType::kHALF ||
           in_out[0].type == nvinfer1::DataType::kFLOAT;
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
      const nvinfer1::PluginFieldCollection*) noexcept -> nvinfer1::IPluginV2*
  {
    auto plugin = new YoloPlugin;
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

  //! TODO
  auto YoloPluginCreator::deserializePlugin(
      [[maybe_unused]] const nvinfer1::AsciiChar* name,
      [[maybe_unused]] const void* serial_data,
      [[maybe_unused]] const size_t serial_length) noexcept
      -> nvinfer1::IPluginV2*

  {
    return nullptr;
  }

}  // namespace DO::Sara::TensorRT
