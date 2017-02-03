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

  //! @brief Iterator class for image dataset.
  class ImageDataSetIterator
  {
  public:
    using self_type = ImageDataSetIterator;
    using file_iterator = std::vector<std::string>::const_iterator;
    using value_type = Image<Rgb8>;

    inline ImageDataSetIterator() = default;

    inline ImageDataSetIterator(file_iterator f)
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

    inline auto operator->() -> const value_type *
    {
      if (_file_i != _file_read)
      {
        imread(_image_read, *_file_i);
        _file_read = _file_i;
      }
      return &_image_read;
    }

    inline auto operator*() -> const value_type&
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

    inline auto path() const -> const std::string&
    {
      return *_file_i;
    }

  private:
    file_iterator _file_i;
    file_iterator _file_read;
    value_type _image_read;
  };


  //! @brief Image dataset class.
  class ImageDataSet
  {
  public:
    using iterator = ImageDataSetIterator;
    using value_type = ImageDataSetIterator::value_type;

    inline ImageDataSet(std::vector<std::string>& image_filepaths)
      : _image_filepaths(image_filepaths)
    {
    }

    void read_from_csv(const std::string& text_filepath);

    inline auto begin() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_image_filepaths.begin()};
    }

    inline auto end() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_image_filepaths.end()};
    }

    inline auto operator[](std::ptrdiff_t i) const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_image_filepaths.begin() + i};
    }

  private:
    std::vector<std::string>& _image_filepaths;
  };


} /* namespace Sara */
} /* namespace DO */
