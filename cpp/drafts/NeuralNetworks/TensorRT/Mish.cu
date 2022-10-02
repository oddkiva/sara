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

#include <drafts/NeuralNetworks/TensorRT/IO.hpp>
#include <drafts/NeuralNetworks/TensorRT/Mish.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <cassert>
#include <stdexcept>


namespace DO::Sara::TensorRT {

  __global__ void mish_kernel(const float* in, float* out,  //
                              const int inout_size)
  {
    // Bound checks.
    const auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= inout_size)
      return;

    const auto v = in[i];
#ifdef USE_FAST_MATH_VERSION
    static constexpr auto thres = 20.f;
    const auto softplus = v > thres    ? v
                          : v < -thres ? __expf(v)
                                       : __logf(1 + __expf(v));
#else
    const auto softplus = logf(1 + expf(v));
#endif
    out[i] = v * tanhf(softplus);
  }


  auto MishPlugin::getOutputDataType(
      [[maybe_unused]] const std::int32_t output_index,
      [[maybe_unused]] const nvinfer1::DataType* input_types,
      [[maybe_unused]] const std::int32_t num_inputs) const noexcept
      -> nvinfer1::DataType
  {
    return nvinfer1::DataType::kFLOAT;  // input_types[0];
  }

  auto MishPlugin::isOutputBroadcastAcrossBatch(
      [[maybe_unused]] const std::int32_t output_index,
      [[maybe_unused]] const bool* input_is_broadcasted,
      [[maybe_unused]] const std::int32_t num_inputs) const noexcept -> bool
  {
    return false;
  }

  auto MishPlugin::canBroadcastInputAcrossBatch(
      [[maybe_unused]] const std::int32_t input_index) const noexcept -> bool
  {
    return false;
  }

  auto MishPlugin::clone() const noexcept -> nvinfer1::IPluginV2Ext*
  {
    try
    {
      auto plugin = new MishPlugin{_inout_size};
      plugin->setPluginNamespace(_namespace.c_str());
      return plugin;
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << "EXCEPTION" << e.what() << std::endl;
    }

    return nullptr;
  }

  auto MishPlugin::getPluginType() const noexcept -> const nvinfer1::AsciiChar*
  {
    return name;
  }

  auto MishPlugin::getPluginVersion() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return version;
  }

  auto MishPlugin::getNbOutputs() const noexcept -> std::int32_t
  {
    return 1;
  }

  auto MishPlugin::getOutputDimensions(
      [[maybe_unused]] const std::int32_t index,  //
      const nvinfer1::Dims* inputs,
      [[maybe_unused]] const std::int32_t nb_input_dims) noexcept
      -> nvinfer1::Dims
  {
    return inputs[0];
  }

  auto MishPlugin::initialize() noexcept -> std::int32_t
  {
    return 0;
  }

  auto MishPlugin::terminate() noexcept -> void
  {
  }

  auto MishPlugin::getWorkspaceSize(
      const std::int32_t /* max_batch_size */) const noexcept -> std::size_t
  {
    return 0;
  }

  auto MishPlugin::enqueue([[maybe_unused]] const std::int32_t batch_size,
                           void const* const* inputs, void* const* outputs,
                           [[maybe_unused]] void* workspace,
                           cudaStream_t stream) noexcept -> std::int32_t
  {
    try
    {
      const auto in = reinterpret_cast<const float*>(inputs[0]);
      const auto out = reinterpret_cast<float*>(outputs[0]);

      // By design CUDA can have at most 1024 threads per block, so let us use
      // this limit.
      static constexpr auto max_threads_per_block = 1024;
      const auto num_blocks = _inout_size % 1024 == 0
                                  ? _inout_size / max_threads_per_block
                                  : (_inout_size + 1) / max_threads_per_block;

      mish_kernel<<<num_blocks, max_threads_per_block, 0, stream>>>(
          in, out, _inout_size);

      return 0;
    }
    catch (const std::exception& e)
    {
      SARA_DEBUG << e.what() << std::endl;
    }

    return -1;
  }

  auto MishPlugin::getSerializationSize() const noexcept -> size_t
  {
    return sizeof(_inout_size);
  }

  auto MishPlugin::serialize(void* buffer) const noexcept -> void
  {
    auto cbuf = reinterpret_cast<char*>(buffer);
    write_to_buffer(cbuf, _inout_size);
  }

  auto MishPlugin::destroy() noexcept -> void
  {
    delete this;
  }

  auto MishPlugin::setPluginNamespace(
      const nvinfer1::AsciiChar* plugin_namespace) noexcept -> void
  {
    _namespace = plugin_namespace;
  }

  auto MishPlugin::getPluginNamespace() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return _namespace.c_str();
  }

  //! TODO
  auto MishPlugin::configurePlugin(
      [[maybe_unused]] const nvinfer1::PluginTensorDesc* inputs,
      [[maybe_unused]] const std::int32_t num_inputs,
      [[maybe_unused]] const nvinfer1::PluginTensorDesc* outputs,
      [[maybe_unused]] const std::int32_t num_outputs) noexcept -> void
  {
  }

  auto MishPlugin::supportsFormatCombination(
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


  MishPluginCreator::MishPluginCreator()
  {
    _plugin_attributes.reserve(1u);
    _plugin_attributes.emplace_back("inout_size", nullptr,
                                    nvinfer1::PluginFieldType::kINT32, 1);
  }

  auto MishPluginCreator::getPluginName() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return MishPlugin::name;
  }

  auto MishPluginCreator::getPluginVersion() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return MishPlugin::version;
  }

  auto MishPluginCreator::getFieldNames() noexcept
      -> const nvinfer1::PluginFieldCollection*
  {
    return &_fc;
  }

  auto MishPluginCreator::createPlugin(
      const nvinfer1::AsciiChar* trt_namespace,
      const nvinfer1::PluginFieldCollection* fc) noexcept
      -> nvinfer1::IPluginV2*
  {
    const auto fields = fc->fields;
    const auto inout_size =
        *reinterpret_cast<const std::int32_t*>(fields[0].data);

    auto plugin = new MishPlugin{inout_size};
    plugin->setPluginNamespace(trt_namespace);
    return plugin;
  }

  auto MishPluginCreator::getPluginNamespace() const noexcept
      -> const nvinfer1::AsciiChar*
  {
    return _namespace.c_str();
  }

  auto MishPluginCreator::setPluginNamespace(
      const nvinfer1::AsciiChar* plugin_namespace) noexcept -> void
  {
    _namespace = plugin_namespace;
  }

  auto MishPluginCreator::deserializePlugin(
      const nvinfer1::AsciiChar* plugin_namespace,  //
      [[maybe_unused]] const void* serial_data,
      [[maybe_unused]] const size_t serial_length) noexcept
      -> nvinfer1::IPluginV2*
  {
    try
    {
      auto buffer_ptr = reinterpret_cast<const char*>(serial_data);
      const auto inout_size = read_from_buffer<std::int32_t>(buffer_ptr);
      auto plugin = new MishPlugin{inout_size};
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
