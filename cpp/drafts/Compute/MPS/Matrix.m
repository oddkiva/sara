#import "Matrix.hpp"


@implementation SaraMatrix

- (instancetype) initWithDevice:(id<MTLDevice>)device
                           rows:(int)rows
                           cols:(int)cols
                        options:(MTLResourceOptions)options
{
  self.buffer = [device newBufferWithLength:rows * cols * sizeof(float) * sizeof(float)
                                    options:options];
  self.desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                    columns:cols
                                                   rowBytes:cols * sizeof(float)
                                                   dataType:MPSDataTypeFloat32];
  self.mat =
    [[MPSMatrix alloc] initWithBuffer:self.buffer descriptor:self.desc];
  return self;
}

- (instancetype) setZero
{
  float* dataPtr = reinterpret_cast<float*>(self.buffer.contents);
  for (unsigned long i = 0; i < self.buffer.length / sizeof(float); ++i)
    dataPtr[i] = 0;

  return self;
}

- (instancetype) setIdentity
{
  float* dataPtr = reinterpret_cast<float*>(self.buffer.contents);
  const unsigned long padded_rows = self.mat.rowBytes / sizeof(float);

  for (unsigned long i = 0; i < self.buffer.length / sizeof(float); ++i)
    dataPtr[i] = 0;

  for (unsigned long i = 0; i < self.mat.rows; ++i)
    dataPtr[i * padded_rows + i] = 1;

  return self;
}

@end
