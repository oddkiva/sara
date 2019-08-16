// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Match.hpp>


namespace DO::Sara {

//! @{
//! @brief Keypoint matching.
auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2,
           float lowe_ratio = 0.6f)
    -> std::vector<Match>;

auto match_keypoints(const std::string& dirpath, const std::string& h5_filepath,
                     bool overwrite) -> void;
//! @}

} /* namespace DO::Sara */
