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

#import "Matrix.hpp"


@implementation SaraMatrix

- (instancetype) initWithDevice : (id<MTLDevice>) device
                           rows : (int) rows
                           cols : (int) cols
                        options : (MTLResourceOptions) options
{
  self.buffer = [device
    newBufferWithLength : rows * cols * sizeof(float) * sizeof(float)
                options : options];
  self.desc = [MPSMatrixDescriptor
    matrixDescriptorWithRows : rows
                     columns : cols
                    rowBytes : cols * sizeof(float)
                    dataType : MPSDataTypeFloat32];
  self.mat =
    [[MPSMatrix alloc] initWithBuffer : self.buffer descriptor : self.desc];

  return self;
}

#if ! __has_feature(objc_arc)
#error "TODO: ARC must be enabled: otherwise fix this!"
- (void) dealloc
{
  [self.buffer release];
  [self.desc release];
  [self.mat release];
  [super dealloc];
}
#endif

- (instancetype) setZero
{
  float* dataPtr = (float*) self.buffer.contents;
  for (unsigned long i = 0; i < self.buffer.length / sizeof(float); ++i)
    dataPtr[i] = 0;

  return self;
}

- (instancetype) setIdentity
{
  float* dataPtr = (float*) self.buffer.contents;
  const unsigned long padded_rows = self.mat.rowBytes / sizeof(float);

  for (unsigned long i = 0; i < self.buffer.length / sizeof(float); ++i)
    dataPtr[i] = 0;

  for (unsigned long i = 0; i < self.mat.rows; ++i)
    dataPtr[i * padded_rows + i] = 1;

  return self;
}

@end
