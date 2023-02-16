// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <utility>


namespace DO::Sara {

  //! @brief Feature global ID (GID).
  struct FeatureGID
  {
    int camera_view_id{-1};
    int feature_index{-1};

    auto operator==(const FeatureGID& other) const -> bool
    {
      return camera_view_id == other.camera_view_id &&
             feature_index == other.feature_index;
    }

    auto operator<(const FeatureGID& other) const -> bool
    {
      return std::make_pair(camera_view_id, feature_index) <
             std::make_pair(other.camera_view_id, other.feature_index);
    }
  };

}  // namespace DO::Sara
