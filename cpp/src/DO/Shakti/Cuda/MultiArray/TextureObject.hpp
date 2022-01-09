// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Shakti/Cuda/MultiArray/TextureArray.hpp>


namespace DO::Shakti::Cuda {

  //! @brief Wrapper class of the Texture object.
  class TextureObject
  {
  public:
    inline TextureObject() = default;

    inline TextureObject(cudaArray_t array,
                         const cudaTextureDesc& texture_descriptor)
      : _array{array}
    {
      auto resource_descriptor = cudaResourceDesc{};
      memset(&resource_descriptor, 0, sizeof(resource_descriptor));
      resource_descriptor.resType = cudaResourceTypeArray;
      resource_descriptor.res.array.array = _array;

      cudaCreateTextureObject(&_texture_object, &resource_descriptor,
                              &texture_descriptor, nullptr);
    }

    inline ~TextureObject()
    {
      if (_texture_object != 0)
      {
        SHAKTI_SAFE_CUDA_CALL(cudaDestroyTextureObject(_texture_object));
        _texture_object = 0;
      }
    }

    inline operator cudaTextureObject_t() const noexcept
    {
      return _texture_object;
    }

  protected:
    cudaArray_t _array = nullptr;
    cudaTextureObject_t _texture_object = 0;
  };

}  // namespace DO::Shakti::Cuda
