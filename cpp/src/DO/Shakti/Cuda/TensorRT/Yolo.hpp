// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <string>
#include <vector>


namespace DO::Sara::TensorRT {

  class YoloPlugin : public nvinfer1::IPluginV2IOExt
  {
  public:
    static constexpr const nvinfer1::AsciiChar* name = "TensorRT-Yolo";
    static constexpr const nvinfer1::AsciiChar* version = "0.1";

    YoloPlugin() = default;

    YoloPlugin(const std::int32_t num_boxes_per_grid_cell,  //
               const std::int32_t num_classes,              //
               const std::int32_t h, const std::int32_t w,  //
               const float scale_x_y)
      : _num_boxes_per_grid_cell{num_boxes_per_grid_cell}
      , _num_classes{num_classes}
      , _h{h}
      , _w{w}
      , _scale_x_y{scale_x_y}
    {
    }

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

    auto setPluginNamespace(const nvinfer1::AsciiChar*) noexcept
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
    //! @brief Number of boxes predicted per grid cell.
    std::int32_t _num_boxes_per_grid_cell;
    //! @brief Number of object classes.
    std::int32_t _num_classes;
    //! @brief Grid dimensions
    //! @{
    std::int32_t _h;
    std::int32_t _w;
    //! @}
    //! @brief Extra parameter for YOLO V4 Tiny.
    float _scale_x_y;

    //! @brief Plugin namespace.
    std::string _namespace;
  };


  class YoloPluginCreator : public nvinfer1::IPluginCreator
  {
  public:
    YoloPluginCreator();

    ~YoloPluginCreator() override = default;

    auto getPluginName() const noexcept -> const nvinfer1::AsciiChar* override;

    auto getPluginVersion() const noexcept
        -> const nvinfer1::AsciiChar* override;

    auto getFieldNames() noexcept
        -> const nvinfer1::PluginFieldCollection* override;

    // N.B.: the plugin namespace should be blank if it is registered
    // statically with the macro REGISTER_TENSORRT_PLUGIN.
    auto createPlugin(const nvinfer1::AsciiChar* plugin_namespace,
                      const nvinfer1::PluginFieldCollection* fc) noexcept
        -> nvinfer1::IPluginV2* override;

    auto getPluginNamespace() const noexcept
        -> const nvinfer1::AsciiChar* override;

    auto setPluginNamespace(const nvinfer1::AsciiChar*) noexcept
        -> void override;

    auto deserializePlugin(const nvinfer1::AsciiChar* plugin_namespace,
                           const void* serial_data,
                           const size_t serial_length) noexcept
        -> nvinfer1::IPluginV2* override;

  private:
    //! @brief Plugin parameters.
    //! N.B.: don't follow example codes where plugin field collection are
    //! declared as static variables. The address sanitizer says it leads to
    //! memory leak otherwise.
    //!
    //! @{
    nvinfer1::PluginFieldCollection _fc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    //! @}

    //! @brief Plugin namespace.
    std::string _namespace;
  };


  REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

}  // namespace DO::Sara::TensorRT
