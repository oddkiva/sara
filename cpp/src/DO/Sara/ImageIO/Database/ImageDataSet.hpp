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

#include <vector>

#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/ImageIO.hpp>


namespace DO { namespace Sara {

  //! @brief Iterator class for image dataset.
  template <typename _Image>
  class ImageDataSetIterator;

  template <typename T>
  class ImageDataSetIterator<Image<T>>
  {
  public:
    using self_type = ImageDataSetIterator;
    using file_iterator = std::vector<std::string>::const_iterator;
    using value_type = Image<T>;

    inline ImageDataSetIterator() = default;

    inline ImageDataSetIterator(file_iterator f, file_iterator f_read)
      : _file_i{f}
      , _file_read{f_read}
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

    inline auto operator->() -> const value_type *
    {
      if (_file_i != _file_read)
      {
        _image_read =  imread<T>(*_file_i);
        _file_read = _file_i;
      }
      return &_image_read;
    }

    inline auto operator*() -> const value_type&
    {
      if (_file_i != _file_read)
      {
        _image_read =  imread<T>(*_file_i);
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

    inline auto path() const -> const std::string&
    {
      return *_file_i;
    }

    inline std::ptrdiff_t operator-(const self_type& other) const
    {
      return _file_i - other._file_i;
    }

  private:
    file_iterator _file_i;
    file_iterator _file_read;
    value_type _image_read;
  };


  //! @brief Image dataset class.
  template <typename _Image>
  class ImageDataSet
  {
  public:
    using container_type = std::vector<std::string>;
    using iterator = ImageDataSetIterator<_Image>;
    using value_type = typename iterator::value_type;

    inline ImageDataSet(std::vector<std::string> image_filepaths)
      : _image_filepaths{std::move(image_filepaths)}
    {
    }

    inline auto begin() const -> iterator
    {
      return iterator{_image_filepaths.begin(), _image_filepaths.end()};
    }

    inline auto end() const -> iterator
    {
      return iterator{_image_filepaths.end(), _image_filepaths.end() };
    }

    inline auto operator[](std::ptrdiff_t i) const -> iterator
    {
      return iterator{_image_filepaths.begin() + i, _image_filepaths.end()};
    }

  private:
    container_type _image_filepaths;
  };


} /* namespace Sara */
} /* namespace DO */
