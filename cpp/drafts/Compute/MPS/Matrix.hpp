#ifndef MATRIX_HPP
#define MATRIX_HPP

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>


@interface SaraMatrix : NSObject
@property(assign, readwrite) id<MTLBuffer> buffer;
@property(assign, readwrite) MPSMatrixDescriptor* desc;
@property(assign, readwrite) MPSMatrix* mat;

- (instancetype) initWithDevice:(id<MTLDevice>)device
                           rows:(int)rows
                           cols:(int)cols
                        options:(MTLResourceOptions)options;
- (instancetype) setZero;
- (instancetype) setIdentity;
@end

#endif // MATRIX_HPP
