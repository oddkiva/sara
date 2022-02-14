#pragma once

#include "OpenCVInterop.hpp"


auto estimate_H(const DO::Sara::OpenCV::Chessboard& chessboard)
    -> Eigen::Matrix3d;
