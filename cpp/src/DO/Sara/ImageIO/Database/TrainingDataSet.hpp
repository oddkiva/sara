// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <tuple>

#include <DO/Sara/ImageIO/Database/ImageDataSet.hpp>


namespace DO { namespace Sara {


  template <typename XIterator, typename YIterator>
  class TrainingDataSetIterator
  {
  public:
    using x_iterator = XIterator;
    using y_iterator = YIterator;

    using x_type = typename XIterator::value_type;
    using y_type = typename YIterator::value_type;
    using self_type = TrainingDataSetIterator;

    inline TrainingDataSetIterator() = default;

    inline TrainingDataSetIterator(x_iterator x, y_iterator y)
      : _x{x}
      , _y{y}
    {
    }

    inline auto operator++() -> self_type&
    {
      ++_x;
      ++_y;
      return *this;
    }

    inline auto operator--() -> self_type&
    {
      --_x;
      --_y;
      return *this;
    }

    inline auto operator+=(std::ptrdiff_t n) -> self_type&
    {
      _x += n;
      _y += n;
      return *this;
    }

    inline auto operator-=(std::ptrdiff_t n) -> self_type&
    {
      _x -= n;
      _y -= n;
      return *this;
    }

    inline auto x() -> const x_type&
    {
      return *_x;
    }

    inline auto y() -> const y_type&
    {
      return *_y;
    }

    inline auto operator==(const self_type& other) const -> bool
    {
      return std::make_pair(_x, _y) == std::make_pair(other._x, other._y);
    }

    inline auto operator!=(const self_type& other) const -> bool
    {
      return !(operator==(other));
    }

  private:
    x_iterator _x;
    y_iterator _y;
  };


  template <typename XHandle, typename YHandle>
  class TrainingDataSet
  {
  public:
    using x_handle = XHandle;
    using y_handle = YHandle;

    using x_set_type = std::vector<XHandle>;
    using y_set_type = std::vector<YHandle>;

    inline TrainingDataSet() = default;

  protected:
    x_set_type _x;
    y_set_type _y;
  };


  class ImageClassificationTrainingDataSet
      : public TrainingDataSet<std::string, int>
  {
    using base_type = TrainingDataSet<std::string, int>;

  public:
    using x_iterator = ImageDataSetIterator;
    using y_iterator = typename y_set_type::const_iterator;

    using iterator = TrainingDataSetIterator<x_iterator, y_iterator>;

    inline ImageClassificationTrainingDataSet() = default;

    void read_from_csv(const std::string& csv_filepath);

    void set_image_data_set(std::vector<std::string> image_filepaths)
    {
      _x = std::move(image_filepaths);
    }

    void set_label_set(std::vector<int> labels)
    {
      _y = std::move(labels);
    }

    inline auto begin() const -> iterator
    {
      return iterator{x_begin(), y_begin()};
    }

    inline auto end() const -> iterator
    {
      return iterator{x_end(), y_end()};
    }

    auto x_begin() const -> x_iterator
    {
      return x_iterator{_x.begin()};
    }

    auto x_end() const -> x_iterator
    {
      return x_iterator{_x.end()};
    }

    auto y_begin() const -> y_iterator
    {
      return _y.begin();
    }

    auto y_end() const -> y_iterator
    {
      return _y.end();
    }
  };


  class ImageSegmentationTrainingDataSet
      : public TrainingDataSet<std::string, std::string>
  {
    using base_type = TrainingDataSet<std::string, std::string>;

  public:
    using x_iterator = ImageDataSetIterator;
    using y_iterator = ImageDataSetIterator;
    using iterator = TrainingDataSetIterator<x_iterator, y_iterator>;

    inline ImageSegmentationTrainingDataSet() = default;

    void read_from_csv(const std::string& csv_filepath);

    inline auto begin() const -> iterator
    {
      return iterator{x_begin(), y_begin()};
    }

    inline auto end() const -> iterator
    {
      return iterator{x_end(), y_end()};
    }

    auto x_begin() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_x.begin()};
    }

    auto x_end() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_x.end()};
    }

    auto y_begin() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_y.begin()};
    }

    auto y_end() const -> ImageDataSetIterator
    {
      return ImageDataSetIterator{_y.end()};
    }
  };

} /* namespace Sara */
} /* namespace DO */
