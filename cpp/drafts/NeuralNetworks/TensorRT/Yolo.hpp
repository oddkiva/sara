#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <vector>


namespace DO::Sara::TensorRT {

  class YoloPlugin : public nvinfer1::IPluginV2IOExt
  {
    auto getOutputDataType(const std::int32_t output_index,
                           const nvinfer1::DataType* input_types,
                           const std::int32_t num_inputs) const noexcept
        -> nvinfer1::DataType override;

    auto
    isOutputBroadcastAcrossBatch(const std::int32_t output_index,  //
                                 const bool* input_is_broadcasted,
                                 const std::int32_t num_inputs) const noexcept
        -> bool override;

    auto
    canBroadcastInputAcrossBatch(const std::int32_t input_index) const noexcept
        -> bool override;

    auto clone() const noexcept -> nvinfer1::IPluginV2Ext* override;

    auto getPluginType() const noexcept -> const nvinfer1::AsciiChar* override;

    auto getPluginVersion() const noexcept
        -> const nvinfer1::AsciiChar* override;

    auto getNbOutputs() const noexcept -> std::int32_t override;

    auto getOutputDimensions(const std::int32_t index,
                             const nvinfer1::Dims* inputs,
                             const std::int32_t nb_input_dims) noexcept
        -> nvinfer1::Dims override;

    auto initialize() noexcept -> std::int32_t override;

    auto terminate() noexcept -> void override;

    auto getWorkspaceSize(std::int32_t max_batch_size) const noexcept
        -> std::size_t override;

    auto enqueue(int32_t batchSize, void const* const* inputs,
                 void* const* outputs, void* workspace,
                 cudaStream_t stream) noexcept -> std::int32_t override;

    auto getSerializationSize() const noexcept -> size_t override;

    auto serialize(void* buffer) const noexcept -> void override;

    auto destroy() noexcept -> void override;

    auto
    setPluginNamespace(const nvinfer1::AsciiChar* plugin_namespace) noexcept
        -> void override;

    auto getPluginNamespace() const noexcept
        -> const nvinfer1::AsciiChar* override;

    auto configurePlugin(const nvinfer1::PluginTensorDesc* inputs,
                         const std::int32_t num_inputs,
                         const nvinfer1::PluginTensorDesc* outputs,
                         const std::int32_t num_outputs) noexcept
        -> void override;

    auto supportsFormatCombination(const std::int32_t pos,  //
                                   const nvinfer1::PluginTensorDesc* in_out,
                                   const std::int32_t nb_inputs,
                                   const std::int32_t nb_outputs) const noexcept
        -> bool override;

  private:
    std::int32_t _classes;
    std::vector<std::int32_t> _anchors;
    std::vector<std::int32_t> _mask;
  };


  class YoloPluginCreator : public nvinfer1::IPluginCreator
  {
    auto getPluginName() const noexcept -> const nvinfer1::AsciiChar* override;

    auto getPluginVersion() const noexcept
        -> const nvinfer1::AsciiChar* override;

    auto getFieldNames() noexcept
        -> const nvinfer1::PluginFieldCollection* override;

    auto createPlugin(const nvinfer1::AsciiChar*,
                      const nvinfer1::PluginFieldCollection*) noexcept
        -> nvinfer1::IPluginV2* override;

    auto getPluginNamespace() const noexcept -> const nvinfer1::AsciiChar* override;

    auto setPluginNamespace(const nvinfer1::AsciiChar* plugin_namespace) noexcept
        -> void override;

    auto deserializePlugin(const nvinfer1::AsciiChar* name,
                           const void* serial_data,
                           const size_t serial_length) noexcept
        -> nvinfer1::IPluginV2* override;

  private:
    static nvinfer1::PluginFieldCollection _fc;
  };

}  // namespace DO::Sara::TensorRT
