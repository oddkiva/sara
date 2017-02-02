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

#include <DO/Sara/ImageIO/Pipeline/ImageDatabase.hpp>


namespace DO { namespace Sara {


  template <typename InputIterator, typename LabelIterator>
  struct TrainingSampleTraits;

  template <typename LabelIterator>
  struct TrainingSampleTraits<ImageDatabaseIterator, LabelIterator>
  {
    using input_iterator = ImageDatabaseIterator;
    using label_iterator = LabelIterator;

    using input_type = typename ImageDatabaseIterator::image_type;
    using label_type = LabelIterator;
  };


  template <typename InputIterator = ImageDatabaseIterator,
            typename LabelIterator = int>
  class TrainingSampleIterator
  {
  public:
    using input_iterator = InputIterator;
    using label_iterator = LabelIterator;
    using sample_iterator = std::pair<input_iterator, label_iterator>;
    using self_type = TrainingSampleIterator;

    TrainingSampleIterator() = default;

    inline auto operator++() -> TrainingSampleIterator&
    {
      ++_s.first;
      ++_s.second;
      return *this;
    }

    inline auto operator--() -> TrainingSampleIterator&
    {
      --_s.first;
      --_s.second;
      return *this;
    }

    inline auto operator+=(std::ptrdiff_t n) -> TrainingSampleIterator&
    {
      _s.first += n;
      _s.second += n;
      return *this;
    }

    inline auto operator-=(std::ptrdiff_t n) -> TrainingSampleIterator&
    {
      _s.first -= n;
      _s.second -= n;
      return *this;
    }

    inline auto operator->() const -> sample_iterator *
    {
      return &_s;
    }

    inline auto operator*() const -> const sample_iterator&
    {
      return _s;
    }

    inline auto operator==(const self_type& other) const -> bool
    {
      return _s == other._s;
    }

    inline auto operator!=(const self_type& other) const -> bool
    {
      return !(operator==(other));
    }

  private:
    sample_iterator _s;
  };



} /* namespace Sara */
} /* namespace DO */
