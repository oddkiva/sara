#include <math_constants.h>


using uint8_t = unsigned char;


__device__ auto lanczos(const float* f, const int x, const int a) -> float
{
  if (x == 0)
    return 1;

  if (-a <= x && x < a)
  {
    static constexpr auto pi = CUDART_PI_F;
    const auto af = static_cast<float>(a);
    const auto pi_x = pi * static_cast<float>(x);
    const auto pi_x_div_a = fdividef(pi_x, af);
    const auto num = af * __sinf(pi_x) * __sinf(pi_x_div_a);
    const auto den = pi_x * pi_x;
    return fdividef(num, den);
  }
  else
    return 0.f;
}

__global__ auto from_hwc_uint8_to_chw_float(float* out, const uint8_t* in,
                                            const int w, const int h) -> void
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= w || y >= h)
    return;

  const auto wh = w * h;
  const int gi_out = y * w + x;
  const int gi_in = 3 * (y * w + x);
  // clang-format off
  out[/* 0 *  wh */ + gi_out] = float(in[gi_in + 0]);
  out[/* 1 */ wh    + gi_out] = float(in[gi_in + 1]);
  out[   2 *  wh    + gi_out] = float(in[gi_in + 2]);
  // clang-format on
}

__global__ auto naive_downsample(float* out, const float* in, const int wout,
                                 const int hout, const int win, const int hin)
    -> void
{
  const int xout = blockIdx.x * blockDim.x + threadIdx.x;
  const int yout = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.z * blockDim.z + threadIdx.z;

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

  const int gi_out = c * wout * hout + yout * wout + xout;
  const int gi_in = c * win * hin + yin * win + xin;
  out[gi_out] = in[gi_in];
}

__global__ auto lanczos_downsample_x(float* out, const float* in,
                                     const int wout, const int hout,
                                     const int win, const int hin) -> void
{
  const int xout = blockIdx.x * blockDim.x + threadIdx.x;
  const int yout = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.z * blockDim.z + threadIdx.z;

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

  const int gi_out = c * wout * hout + yout * wout + xout;
  const int gi_in = c * win * hin + yin * win + xin;
  out[gi_out] = in[gi_in];
}
