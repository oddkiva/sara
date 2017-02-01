// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/ImageIO.hpp>


namespace DO { namespace Sara {


  class ImageDatabaseIterator
  {
  public:
    using self_type = ImageDatabaseIterator;
    using file_iterator = std::vector<std::string>::const_iterator;
    using image_type = Image<Rgb8>;

    inline ImageDatabaseIterator() = default;

    inline ImageDatabaseIterator(file_iterator f)
      : _file_i{f}
    {
    }

    inline auto operator++() -> self_type&
    {
      ++_file_i;
      return *this;
    }

    inline auto operator--() -> self_type&
    {
      --_file_i;
      return *this;
    }

    inline auto operator+=(std::ptrdiff_t n) -> self_type&
    {
      _file_i += n;
      return *this;
    }

    inline auto operator-=(std::ptrdiff_t n) -> self_type&
    {
      _file_i -= n;
      return *this;
    }

    inline auto operator+(std::ptrdiff_t n) const -> self_type
    {
      auto it = *this;
      it += n;
      return it;
    }

    inline auto operator-(std::ptrdiff_t n) const -> self_type
    {
      auto it = *this;
      it -= n;
      return it;
    }

    inline auto operator->() -> const image_type *
    {
      if (_file_i != _file_read)
      {
        imread(_image_read, *_file_i);
        _file_read = _file_i;
      }
      return &_image_read;
    }

    inline auto operator*() -> const image_type&
    {
      if (_file_i != _file_read)
      {
        imread(_image_read, *_file_i);
        _file_read = _file_i;
      }
      return _image_read;
    }

    inline auto operator==(const self_type& other) const -> bool
    {
      return _file_i == other._file_i;
    }

    inline auto operator!=(const self_type& other) const -> bool
    {
      return !(operator==(other));
    }

    inline friend auto begin(const std::vector<std::string>& image_filepaths)
        -> self_type
    {
      auto it = self_type{};
      it._file_i = image_filepaths.begin();
      return it;
    }

    inline friend auto end(const std::vector<std::string>& image_filepaths)
        -> self_type 
    {
      return ImageDatabaseIterator{};
    }

  private:
    file_iterator _file_i;
    file_iterator _file_read;
    image_type _image_read;
  };


} /* namespace Sara */
} /* namespace DO */