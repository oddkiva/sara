#pragma once

#include "OpenCVInterop.hpp"


inline auto estimate_H(const DO::Sara::OpenCV::Chessboard& chessboard)
    -> Eigen::Matrix3d;
