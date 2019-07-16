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

  struct DeviceDeleter
  {
    void operator()(const void*) const;
  };

  static std::unique_ptr<void, DeviceDeleter> _device;
};

} /* namespace DO::Sara */

#endif  // SGEMMIMPL_HPP
