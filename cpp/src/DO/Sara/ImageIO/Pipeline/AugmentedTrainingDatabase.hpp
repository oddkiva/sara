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


namespace DO { namespace Sara {

  template <typename X, typename Y>
  class TransformedTrainingSampleIterator : public TrainingSampleIterator<X, Y>
  {
  public:
    using x_type = typename TrainingSampleTraits<X, Y>::input_type;
    using y_type = typename TrainingSampleTraits<X, Y>::label_type;

    auto x() -> x_type
    {
      return x_type{};
    }

    auto y() -> y_type
    {
      return y_type{};
    }

  private:
    TransformParameter _transform_param;
  };

} /* namespace Sara */
} /* namespace DO */
