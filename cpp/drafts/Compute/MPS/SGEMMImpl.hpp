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

#ifndef SGEMMIMPL_HPP
#define SGEMMIMPL_HPP

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <memory>


namespace DO::Sara {

struct SGEMMImpl
{
  SGEMMImpl();

  ~SGEMMImpl();

  void operator()(int m, int n, int k, float alpha, const float* A,
                  const float* B, float beta, float* C) const;
};

} /* namespace DO::Sara */

#endif  // SGEMMIMPL_HPP
