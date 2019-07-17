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

#ifndef MATRIX_HPP
#define MATRIX_HPP

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>


@interface SaraMatrix : NSObject
@property(strong, readwrite) id<MTLBuffer> buffer;
@property(strong, readwrite) MPSMatrixDescriptor* desc;
@property(strong, readwrite) MPSMatrix* mat;

- (instancetype) initWithDevice : (id<MTLDevice>)device
                           rows : (int)rows
                           cols : (int)cols
                        options : (MTLResourceOptions)options;
#if ! __has_feature(objc_arc)
#error "TODO: ARC must be enabled"
- (void) dealloc;
#endif
- (instancetype) setZero;
- (instancetype) setIdentity;
@end

#endif // MATRIX_HPP
