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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include "SGEMM.hpp"

#import "Matrix.hpp"
#import "SGEMMImpl.hpp"

#include <algorithm>
#include <stdexcept>


@interface SGEMMImplDetails : NSObject
@end

std::unique_ptr<void, DO::Sara::SGEMMImpl::DeviceDeleter> DO::Sara::SGEMMImpl::_device{nullptr};

@implementation SGEMMImplDetails
void DO::Sara::SGEMMImpl::DeviceDeleter::operator()(const void* device) const
{
  if (_device)
    [(id) device dealloc];
}


DO::Sara::SGEMMImpl::SGEMMImpl()
{
  if (!_device)
  {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    _device.reset(reinterpret_cast<void *>(d));
  }
}

DO::Sara::SGEMMImpl::~SGEMMImpl()
{
}


void DO::Sara::SGEMMImplDeleter::operator()(DO::Sara::SGEMMImpl *impl) const
{
  delete impl;
}


void DO::Sara::SGEMMImpl::operator()(int m, int n, int k, float alpha, const float* A,
                                     const float* B, float beta, float* C) const
{
  id<MTLDevice> device = (id<MTLDevice>) SGEMMImpl::_device.get();

  id<MTLCommandQueue> commandQueue = [(id) device newCommandQueue];
  if (commandQueue == nil)
    throw std::runtime_error{"Could not get valid command queue!"};

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  if (commandBuffer == nil)
    throw std::runtime_error{"Could not get valid command queue!"};

  SaraMatrix* matA = [[SaraMatrix alloc] initWithDevice : device
                                                   rows : m
                                                   cols : k
                                                options : MTLResourceStorageModeShared];
  std::copy(A, A + m * k, reinterpret_cast<float*>(matA.buffer.contents));

  SaraMatrix* matB = [[SaraMatrix alloc] initWithDevice : device
                                                   rows : k
                                                   cols : n
                                                options : MTLResourceStorageModeShared];
  std::copy(B, B + k * n, reinterpret_cast<float*>(matB.buffer.contents));

  SaraMatrix* matC = [[SaraMatrix alloc] initWithDevice : device
                                                   rows : m
                                                   cols : n
                                                options : MTLResourceStorageModeShared];

  // Get the matmul program written in Metal.
  MPSMatrixMultiplication* sgemm =
    [[MPSMatrixMultiplication alloc] initWithDevice : device
                                      transposeLeft : NO
                                     transposeRight : NO
                                         resultRows : matA.mat.rows
                                      resultColumns : matB.mat.columns
                                    interiorColumns : matA.mat.columns
                                              alpha : (double) alpha
                                               beta : (double) beta];

  // Wrap the Metal program in a GPU operation.
  [sgemm encodeToCommandBuffer : commandBuffer
                    leftMatrix : matA.mat
                   rightMatrix : matB.mat
                  resultMatrix : matC.mat];

  [commandBuffer commit];

  [commandBuffer waitUntilCompleted];

  auto matC_ptr = reinterpret_cast<float*>(matC.buffer.contents);
  std::copy(matC_ptr, matC_ptr + m * n, C);

  [matC dealloc];
  [matB dealloc];
  [matA dealloc];

  [sgemm dealloc];
}
@end
