#pragma once

#include <NvInfer.h>

#include <cassert>
#include <stdexcept>


namespace DO::Sara::TensorRT {

  class YoloPlugin : public nvinfer1::IPluginV2IOExt
  {
    auto getOutputDataType([[maybe_unused]] const std::int32_t output_index,
                           const nvinfer1::DataType* input_types,
                           [[maybe_unused]] const std::int32_t num_inputs)
        const noexcept -> nvinfer1::DataType override
    {
      assert(output_index == 0 && num_inputs == 1);
      return input_types[0];
    }

    auto isOutputBroadcastAcrossBatch(
        [[maybe_unused]] const std::int32_t output_index,
        [[maybe_unused]] const bool* input_is_broadcasted,
        [[maybe_unused]] const std::int32_t num_inputs) const noexcept
        -> bool override
    {
      return false;
    }

    auto canBroadcastInputAcrossBatch(
        [[maybe_unused]] const std::int32_t input_index) const noexcept
        -> bool override
    {
      return false;
    }

    auto clone() const noexcept -> nvinfer1::IPluginV2Ext* override
    {
      return nullptr;
    }

    auto getPluginType() const noexcept -> const nvinfer1::AsciiChar* override
    {
      return "Yolo";
    }

    auto getPluginVersion() const noexcept
        -> const nvinfer1::AsciiChar* override
    {
      return "0.1";
    }

    auto getNbOutputs() const noexcept -> std::int32_t override
    {
      return 1;
    }

    auto getOutputDimensions(
        [[maybe_unused]] const std::int32_t index,
        [[maybe_unused]] const nvinfer1::Dims* inputs,
        [[maybe_unused]] const std::int32_t nb_input_dims) noexcept
        -> nvinfer1::Dims override
    {
      return inputs[0];
    }

    auto initialize() noexcept -> std::int32_t override
    {
      return 1;
    }

    auto terminate() noexcept -> void override
    {
    }


    auto getWorkspaceSize(int32_t maxBatchSize) const noexcept
        -> std::size_t override
    {
      return 1;
    }

    auto enqueue(int32_t batchSize, void const* const* inputs,
                 void* const* outputs, void* workspace,
                 cudaStream_t stream) noexcept -> std::int32_t override
    {
      return 0;
    }

    auto getSerializationSize() const noexcept -> size_t override
    {
      // Find the size of the serialization buffer required. More...
      return 0;
    }

    auto serialize(void* buffer) const noexcept -> void override
    {
      // Serialize the layer.More...
    }

    auto destroy() noexcept -> void override
    {
      // Destroy the plugin object .This will be called when the network,
      // builder or engine is destroyed.More...
    }

    auto setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
        -> void override
    {
      // Set the namespace that this plugin object belongs to.Ideally, all
      // plugin objects from the same plugin library should have the same
      // namespace.More...
    }

    auto getPluginNamespace() const noexcept
        -> const nvinfer1::AsciiChar* override
    {
      // Return the namespace of the plugin object .More...
      return nullptr;
    }

    auto
    configurePlugin([[maybe_unused]] const nvinfer1::PluginTensorDesc* inputs,
                    [[maybe_unused]] const std::int32_t num_inputs,
                    [[maybe_unused]] const nvinfer1::PluginTensorDesc* outputs,
                    [[maybe_unused]] const std::int32_t num_outputs) noexcept
        -> void override
    {
    }

    auto supportsFormatCombination(
        [[maybe_unused]] const std::int32_t pos,  //
        const nvinfer1::PluginTensorDesc* in_out,
        [[maybe_unused]] const std::int32_t nb_inputs,
        [[maybe_unused]] const std::int32_t nb_outputs) const noexcept
        -> bool override
    {
      assert(nb_inputs == 1 || nb_outputs == 1 || pos == 0);

      return in_out[0].type == nvinfer1::DataType::kHALF ||
             in_out[0].type == nvinfer1::DataType::kFLOAT;
    }
  };

  class YoloPluginCreator : public nvinfer1::IPluginCreator
  {
    auto getPluginName() const noexcept -> const nvinfer1::AsciiChar* override
    {
      return "Yolo";
    }

    auto getPluginVersion() const noexcept
        -> const nvinfer1::AsciiChar* override
    {
      return "0.1";
    }

    auto getFieldNames() noexcept
        -> const nvinfer1::PluginFieldCollection* override
    {
      return nullptr;
    }

    auto createPlugin(const nvinfer1::AsciiChar*,
                      const nvinfer1::PluginFieldCollection*) noexcept
        -> nvinfer1::IPluginV2* override
    {
      return nullptr;
    }
  };

}  // namespace DO::Sara::TensorRT
