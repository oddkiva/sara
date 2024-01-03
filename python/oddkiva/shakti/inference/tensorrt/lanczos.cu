#include <math_constants.h>


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

__global__ auto lanczos_x(float* out,
                          const float* in,                 //
                          const int wout, const int hout,  //
                          const int win, const int hin,    //
                          const int a)  // Lanczos band parameter
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
  if (yin >= hin)
    yin = hin - 1;

  const auto gi_out = c * wout * hout + yout * wout + xout;

  const auto clamp_x_coordinate = [win](const int x) {
    return min(max(0, xin + k), w_in - 1);
  };

  // Naive implementation.
  // TODO: use the shared memory technique.
  // __shared__ float xins[32];
  // ...
  // __syncthreads();
  auto val = 0.f;
  for (int k = -a; k <= a; ++k)
  {
    // Clamp the coordinates.
    const auto xin_k = clamp_x_coordinate(xin + k);
    // Get the flattened array index.
    const auto gi_in_k = c * win * hin + yin * win + xin_k;
    val += in[gi_in_k];
  }
  val = fdividef(val, 2 * a + 1);
  out[gi_out] = val;
}
