// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <math_constants.h>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/ImageProcessing/Kernels/Globals.hpp>

#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>

#include <DO/Shakti/Utilities/Timer.hpp>


namespace DO { namespace Shakti {

  __constant__ float bin_scale_unit_length;
  __constant__ float max_bin_value;
  __constant__ float sigma;

  template <int N, int O>
  __device__
  void accumulate(Vector<float, N*N*O>& h, const Vector2f& pos,
                  float ori, float weight, float mag)
  {
    auto at = [](int i, int j, int o) {
      return N*O*i + j*O + o;
    };

    // By trilinear interpolation, we mean that in this translated coordinate
    // frame, a gradient with orientation $\theta$ and located at
    // $(x,y) \in [-1,N]^2$ contributes to the 4 histograms:
    //  - $\mathbf{h}_{ floor(y)  , floor(x)  }$
    //  - $\mathbf{h}_{ floor(y)  , floor(x)+1}$
    //  - $\mathbf{h}_{ floor(y)+1, floor(x)  }$
    //  - $\mathbf{h}_{ floor(y)+1, floor(x)+1}$
    // In each of these histograms, the following bins are accumulated:
    //  - $\mathbf{h}_{o}$
    //  - $\mathbf{h}_{o+1}$
    // where $o = floor(\theta * O/ (2*\pi))$
    //
    // Note that a gradient at the boundary like $(-1,-1)$ contributes only
    // to P_{0,0}.
    float xfrac = pos.x() - floor(pos.x());
    float yfrac = pos.y() - floor(pos.y());
    float orifrac = ori - floor(ori);
    int int_x = int(pos.x());
    int int_y = int(pos.y());
    int int_ori = int(ori);

#pragma unroll
    for (int dy = 0; dy < 2; ++dy)
    {
      int y = int_y + dy;
      if (y < 0 || y >= N)
        continue;
      float wy = (dy == 0) ? 1.f - yfrac : yfrac;

#pragma unroll
      for (int dx = 0; dx < 2; ++dx)
      {
        int x = int_x + dx;
        if (x < 0 || x >= N)
          continue;
        float wx = (dx == 0) ? 1.f - xfrac : xfrac;

#pragma unroll
        for (int dori = 0; dori < 2; ++dori)
        {
          int o = (int_ori + dori) % O;
          float wo = (dori == 0) ? 1.f - orifrac : orifrac;
          // Trilinear interpolation:
          h[at(y, x, o)] += wy*wx*wo*weight*mag;
        }
      }
    }
  }

  //! \brief Robustify descriptor w.r.t. illumination changes.
  template <int N, int O>
  __device__
  void normalize(Vector<float, N*N*O>& h)
  {
    // Euclidean normalization.
    h.normalize();
    // Clamp histogram bin values $h_i$ to 0.2 for enhanced robustness to
    // lighting change.
#pragma unroll
    for (int i = 0; i < N*N*O; ++i)
      h[i] = min(h[i], max_bin_value);
    // Re-normalize again.
    h.normalize();
  }

  template <int N, int O>
  __global__
  void compute_dense_upright_sift_descriptor(Vector<float, N*N*O> *out_histogram)
  {
    enum { Dim = N*N*O };

    const auto i = offset<2>();
    const auto p = coords<2>();

    const auto& x = p.x();
    const auto& y = p.y();
    const auto l = bin_scale_unit_length * sigma;

    const auto pi = float(CUDART_PI);
    const auto r = sqrt(2.f) * bin_scale_unit_length * (N + 1) / 2.f;

    const auto rounded_r = int(r);

    Vector<float, Dim> h;
    for (int v = -rounded_r; v <= rounded_r; ++v)
    {
      for (int u = -rounded_r; u <= rounded_r; ++u)
      {
        const auto weight = exp(-(u*u + v*v) / (2.f*powf(N / 2.f, 2)));
        auto grad = tex2D(in_float2_texture, x + u, y + v);
        auto mag = grad.x;
        auto ori = grad.y;

        ori = ori < 0.f ? ori + 2.f*pi : ori;
        ori *= float(O) / (2.f*pi);

        Vector2f pos{
          u/l + N / 2.f - 0.5f,
          v/l + N / 2.f - 0.5f
        };

        accumulate<N, O>(h, pos, ori, weight, mag);
      }
    }

    if (h.norm() > 1e-6f)
    {
      h.normalize();
      h *= 512.f;
    }

#pragma unroll
    for (int i = 0; i < Dim; ++i)
      h[i] = min(h[i], 255.f);

    out_histogram[i] = h;

  }

  template <int N, int O>
  MultiArray<Vector<float, N*N*O>, 2>
  compute_dense_upright_sift_descriptor(const TextureArray<Vector2f>& gradients)
  {
    MultiArray<Vector<float, N*N*O>, 2> sifts{ gradients.sizes() };
    const auto block_size = default_block_size_2d();
    const auto grid_size = grid_size_2d(sifts);
    SHAKTI_SAFE_CUDA_CALL(cudaBindTextureToArray(in_float2_texture, gradients));
    compute_dense_upright_sift_descriptor<N, O><<<grid_size, block_size>>>(sifts.data());
    SHAKTI_SAFE_CUDA_CALL(cudaUnbindTexture(in_float2_texture));
    return sifts;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  DenseSiftComputer::DenseSiftComputer()
  {
    using namespace std;
    cout << _bin_scale_unit_length << endl;
    cout << _max_bin_value << endl;
    cout << _sigma << endl;

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      bin_scale_unit_length, &_bin_scale_unit_length, sizeof(float)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      max_bin_value, &_max_bin_value, sizeof(float)));
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      sigma, &_sigma, sizeof(float)));
  }

  MultiArray<Vector<float, 128>, 2>
  DenseSiftComputer::operator()(const TextureArray<Vector2f>& gradients) const
  {
    return Shakti::compute_dense_upright_sift_descriptor<4, 8>(gradients);
  }

  void
  DenseSiftComputer::operator()(float *out, const float *in,
                                const int *sizes) const
  {
    // Compute gradients in polar coordinates.
    TextureArray<float> in_cuda_array{ in, sizes };
    MultiArray<Vector2f, 2> gradients_polar_coords{
      gradient_polar_coords(in_cuda_array)
    };

    // Now compute SIFT descriptors densely from the gradients.
    TextureArray<Vector2f> in_gradients_cuda_array{
      gradients_polar_coords.data(), gradients_polar_coords.sizes(),
      cudaMemcpyDeviceToDevice
    };

    auto sifts = this->operator()(in_gradients_cuda_array);
    sifts.copy_to_host(reinterpret_cast<Vector<float, 128> *>(out));
  }

} /* namespace Shakti */
} /* namespace DO */
