#pragma once

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include <drafts/OpenCL/Core/Context.hpp>
#include <drafts/OpenCL/Core/Device.hpp>
#include <drafts/OpenCL/Core/DeviceBuffer.hpp>
#include <drafts/OpenCL/Core/DeviceImage.hpp>
#include <drafts/OpenCL/Core/Error.hpp>
#include <drafts/OpenCL/Core/Kernel.hpp>


namespace DO::Sara {

  class CommandQueue
  {
  public:
    CommandQueue() = default;

    CommandQueue(const Context& context, const Device& device)
    {
      initialize(context, device);
    }

    ~CommandQueue()
    {
      release();
    }

    void initialize(const Context& context, const Device& device)
    {
      auto err = cl_int{};
      _queue = clCreateCommandQueue(context, device, _properties, &err);
      if (err < 0)
        throw std::runtime_error(format(
        "Error: failed to create command queue! %s\n",
        get_error_string(err)));
    }

    void release()
    {
      auto err = clReleaseCommandQueue(_queue);
      if (err < 0)
        throw std::runtime_error(format(
        "Error: failed to release command queue! %s\n",
        get_error_string(err)));
    }

    void finish()
    {
      auto err = clFinish(_queue);
      if (err < 0)
        throw std::runtime_error(format(
        "Error: failed to finish command queue! %s\n",
        get_error_string(err)));
    }

    void enqueue_nd_range_kernel(Kernel& kernel,
                                 cl_uint work_dims,
                                 const size_t *global_work_offsets,
                                 const size_t *global_work_sizes,
                                 const size_t *local_work_sizes)
    {
      auto err = clEnqueueNDRangeKernel(
        _queue, kernel, work_dims,
        global_work_offsets, global_work_sizes, local_work_sizes,
        0, nullptr, nullptr);
      if (err)
        throw std::runtime_error(format(
        "Error: Failed to execute kernel! %s",
        get_error_string(err)));
    }

    template <typename T>
    void enqueue_read_buffer(DeviceBuffer<T>& src, T* dst, bool blocking = true)
    {
      cl_int err = clEnqueueReadBuffer(
        _queue, src, cl_bool(blocking), 0,
        src.size() * sizeof(T), dst, 0, nullptr, nullptr);
      if (err)
        throw std::runtime_error(format(
        "Error: Failed to copy buffer from device to host! %s",
        get_error_string(err)));
    }

    template <typename T>
    void enqueue_read_image(DeviceImage<T, 2>& src, T* dst,
                            bool blocking = true)
    {
      size_t origin[3] = { 0, 0, 0 };
      size_t region[3] = { src.width(), src.height(), 1 };
      
      cl_int err = clEnqueueReadImage(
        _queue, src, cl_bool(blocking),
        origin, region, 0, 0, dst,
        0, nullptr, nullptr);
      if (err)
        throw std::runtime_error(format(
        "Error: Failed to copy buffer from device to host! %s",
        get_error_string(err)));
    }

    operator cl_command_queue() const
    {
      return _queue;
    }

  private:
    cl_command_queue _queue = nullptr;
    cl_command_queue_properties _properties = 0;
  };

} /* namespace DO::Sara */
