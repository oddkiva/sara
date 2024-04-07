#include "downsample.hpp"

__global__ auto
naive_downsample_and_transpose_impl(float* out_chw, const std::uint8_t* in_hwc,
                                    const int wout, const int hout,
                                    const int win, const int hin) -> void
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int yout = blockIdx.y * blockDim.y + threadIdx.y;
  const int xout = blockIdx.z * blockDim.z + threadIdx.z;

  if (xout >= wout || yout >= hout || c >= 3)
    return;

  const float sx = float(win) / float(wout);
  const float sy = float(hin) / float(hout);

  int xin = int(xout * sx + 0.5f);
  int yin = int(yout * sy + 0.5f);

  if (xin >= win)
    xin = win - 1;
  if (yin >= hin)
    yin = hin - 1;

  const int gi_out = c * hout * wout + yout * wout + xout;
  const int gi_in = yin * win * 3 + xin * 3 + c;

  static constexpr auto normalize_factor = 1 / 255.f;
  out_chw[gi_out] = static_cast<float>(in_hwc[gi_in]) * normalize_factor;
}

auto naive_downsample_and_transpose(CudaManagedTensor3f& tensor_chw_resized_32f,
                                    const CudaManagedTensor3ub& tensor_hwc_8u)
    -> void
{
  // Data order: H W C
  //             0 1 2
  const auto in_hwc = tensor_hwc_8u.data();
  const auto win = tensor_hwc_8u.sizes()(1);
  const auto hin = tensor_hwc_8u.sizes()(0);

  // Data order: C H W
  //             0 1 2
  auto out_chw = tensor_chw_resized_32f.data();
  const auto hout = tensor_chw_resized_32f.sizes()(1);
  const auto wout = tensor_chw_resized_32f.sizes()(2);

  static const auto threads_per_block = dim3{4, 16, 16};
  static const auto num_blocks = dim3{
      1,  //
      (hout + threads_per_block.y - 1) / threads_per_block.y,
      (wout + threads_per_block.z - 1) / threads_per_block.z  //
  };

  naive_downsample_and_transpose_impl<<<num_blocks, threads_per_block>>>(
      out_chw, in_hwc,  //
      wout, hout,       //
      win, hin          //
  );
}