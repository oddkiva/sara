// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>


using namespace std;


namespace DO { namespace Sara {

  bool on_edge(const ImageView<float>& I, int x, int y, float edge_ratio)
  {
    auto H = hessian(I, Point2i{x, y});
    return pow(H.trace(), 2)*edge_ratio >=
           pow(edge_ratio+1.f, 2)*fabs(H.determinant());
  }

  bool refine_extremum(const ImagePyramid<float>& I, int x, int y, int s, int o,
                       int type, Point3f& pos, float& val, int border_sz,
                       int num_iter)
  {
    Vector3f D_prime; // gradient
    Matrix3f D_second; // hessian
    Vector3f h;
    Vector3f lambda;

    pos << float(x), float(y), I.scale_relative_to_octave(s);

    int i;
    for (i = 0; i < num_iter; ++i)
    {
      // Range check at each iteration. The first iteration should always be OK.
      if (x < border_sz || x >= I(s,o).width() - border_sz  ||
          y < border_sz || y >= I(s,o).height() - border_sz ||
          s < 1 || s >= static_cast<int>(I(o).size()) - 1)
        break;

      // Estimate the gradient and the hessian matrix by central finite
      // differentiation.
      D_prime = gradient(I,x,y,s,o);
      D_second = hessian(I,x,y,s,o);

      // The interpolation or refinement is done conservatively depending on the
      // quality of the Hessian matrix estimate.
      //
      // If $(x,y,s,o)$ is a local maximum ('type == 1'),
      //   the Hessian matrix $H$ must be ***negative-definite***.
      //
      // If $(x,y,s,o)$ is a local minimum ('type == -1'),
      //   the Hessian matrix $H$ must be ***positive-definite***.
      //
      // Otherwise:
      // - either we are localizing a saddle point instead of an
      //   extremum if $D''(\mathbf{x})$ is invertible;
      // - or Newton's method is applicable if $D''(\mathbf{x})$ is not
      //   invertible.
      // Such case arises frequently and in that case, interpolation is not
      // done.
      SelfAdjointEigenSolver<Matrix3f> solver(D_second);
      lambda = solver.eigenvalues();
      // Not sure about numerical errors... But it might just work out for now.
      if ((lambda*float(type)).maxCoeff() >= 0)
      {
        h.setZero();
        break;
      }

      // $D''(\mathbf{x})$ is just a 3x3 matrix and computing its inverse is
      // thus cheap (cf. Eigen library.).
      h = - D_second.inverse() * D_prime;

      // The interpolated extremum should be normally close to the initial
      // position which has integral coordinates. Otherwise, the estimates of
      // the gradient and the Hessian matrix are bad.
      if (h.block(0,0,2,1).cwiseAbs().maxCoeff() > 1.5f)
      {
        //#define VERBOSE
  #ifdef VERBOSE
        print_stage("Offset is too large: don't refine");
        cout << "offset = " << h.transpose() << endl;
  #endif
        return false;
      }

      // Contrary to what is said in the paper, Lowe's implementation
      // refines iteratively the position of extremum only w.r.t spatial
      // variables $x$ and $y$ while the scale variable $\sigma$ which is
      // updated only once.
      if (h.block(0,0,2,1).cwiseAbs().minCoeff() > 0.6f)
      {
        x += h(0) > 0 ? 1 : -1;
        y += h(1) > 0 ? 1 : -1;
        continue;
      }

      // Stop iterating.
      break;
    }

    pos = Vector3f{ float(x), float(y), I.scale_relative_to_octave(s) };
    float oldval = I(x,y,s,o);
    float newval = oldval + 0.5f*D_prime.dot(h);

    if ( (type==1 && oldval <= newval) || (type==-1 && oldval >= newval) )
    {
      pos += h;
      val = newval;
    }
    else
      std::cerr << "INTERPOLATION ERROR" << std::endl;

    return true;
  }

  bool refine_extremum(const ImageView<float>& I, int x, int y, int type,
                       Point2f& pos, float& val, int border_sz, int num_iter)
  {
    auto D_prime = Vector2f{}; // gradient
    auto D_second = Matrix2f{}; // hessian
    auto h = Vector2f{}; // offset to estimate

    pos << float(x), float(y);

    int i;
    for (i = 0; i < num_iter; ++i)
    {
      // Range check at each iteration. The first iteration should always be OK.
      if (x < border_sz || x >= I.width() - border_sz  ||
          y < border_sz || y >= I.height() - border_sz )
        break;

      // Estimate the gradient and the hessian matrix by central finite
      // differentiation.
      D_prime = gradient(I, Point2i{ x, y });
      D_second = hessian(I, Point2i{ x, y });

      // The interpolation or refinement is done conservatively depending on the
      // quality of the Hessian matrix estimate.
      //
      // If $(x,y)$ is a local maximum ('type == 1'),
      //   the Hessian matrix $H$ must be ***negative-definite***.
      //
      // If $(x,y)$ is a local minimum ('type == -1'),
      //   the Hessian matrix $H$ must be ***positive-definite***.
      //
      // Otherwise:
      // - either we are localizing a saddle point instead of an
      //   extremum if $D''(\mathbf{x})$ is invertible;
      // - or Newton's method is applicable if $D''(\mathbf{x})$ is not
      //   invertible.
      // Such case arises frequently and in that case, interpolation is not
      // done.
      // We just need to check the determinant and the trace in 2D.
      if (D_second.determinant() <= 0.f || D_second.trace()*type >= 0.f)
      {
        D_prime.setZero();
        break;
      }

      // $D''(\mathbf{x})$ is just a 3x3 matrix and computing its inverse is
      // thus cheap (cf. Eigen library.).
      h = -D_second.inverse() * D_prime;

      // The interpolated extremum should be normally close to the initial
      // position which has integral coordinates. Otherwise, the estimates of
      // the gradient and the Hessian matrix are bad.
      if (h.cwiseAbs().maxCoeff() > 1.5f)
      {
        //#define VERBOSE
#ifdef VERBOSE
        print_stage("Offset is too large: don't refine");
        cout << "offset = " << h.transpose() << endl;
#endif
        return false;
      }

      // Contrary to what is said in the paper, Lowe's implementation
      // refines iteratively the position of extremum only w.r.t spatial
      // variables $x$ and $y$ while the scale variable $\sigma$ which is
      // updated only once.
      if (h.cwiseAbs().minCoeff() > 0.6f)
      {
        x += h(0) > 0 ? 1 : -1;
        y += h(1) > 0 ? 1 : -1;
        continue;
      }
      // Stop iterating.
      break;
    }

    pos << float(x), float(y);
    float oldval = I(x, y);
    float newval = oldval + 0.5f * D_prime.dot(h);

    if ( (type==1 && oldval <= newval) || (type==-1 && oldval >= newval) )
    {
      pos += h;
      val = newval;
    }
    else
      std::cerr << "INTERPOLATION ERROR" << std::endl;

    return true;
  }

  vector<OERegion> local_scale_space_extrema(const ImagePyramid<float>& I,
                                             int s, int o,
                                             float extremum_thres,
                                             float edge_ratio_thres,
                                             int img_padding_sz,
                                             int refine_iterations)
  {
    auto extrema = std::vector<OERegion>{};
    extrema.reserve(10000);

    auto map = Image<int>{ I(s,o).sizes() };
    map.array().setZero();

//#define STRICT_LOCAL_EXTREMA
#ifdef STRICT_LOCAL_EXTREMA
    LocalScaleSpaceExtremum<std::greater, float> local_max;
    LocalScaleSpaceExtremum<std::less, float> local_min;
#else
    LocalScaleSpaceExtremum<std::greater_equal, float> local_max;
    LocalScaleSpaceExtremum<std::less_equal, float> local_min;
#endif

    for (int y = img_padding_sz; y < I(s,o).height() - img_padding_sz; ++y)
    {
      for (int x = img_padding_sz; x < I(s,o).width() - img_padding_sz; ++x)
      {
        // Identify extremum type if it is one
        int type = 0;
        if (local_max(x,y,s,o,I))
          type = 1; // maximum
        else if (local_min(x,y,s,o,I))
          type = -1; // minimum
        else
          continue;
#ifndef STRICT_LOCAL_EXTREMA
        // Reject early.
        if (std::abs(I(x,y,s,o)) < 0.8f*extremum_thres)
          continue;
#endif
        // Reject early if located on edge.
        if (on_edge(I(s,o), x, y, edge_ratio_thres))
          continue;
        // Try to refine extremum.
        auto pos = Point3f{};
        auto val = float{};
        /*if (!refineExtremum(I,x,y,s,o,type,pos,val,img_padding_sz,refine_iterations))
          continue;*/
        refine_extremum(I,x,y,s,o,type,pos,val,img_padding_sz,refine_iterations);

        // Don't add if already marked.
        if (map(static_cast<int>(x), static_cast<int>(y)) == 1)
          continue;
#ifndef STRICT_LOCAL_EXTREMA
        // Reject if contrast too low.
        if (std::abs(val) < extremum_thres)
          continue;
#endif
        // Store the DoG extremum.
        auto dog = OERegion(pos.head<2>(), pos.z());

        dog.extremum_value() = val;
        dog.extremum_type() = type == 1 ?
          OERegion::ExtremumType::Max : OERegion::ExtremumType::Min;
        extrema.push_back(dog);
        map(static_cast<int>(x), static_cast<int>(y)) = 1;
      }
    }

    return extrema;
  }


  bool select_laplace_scale(float& scale,
                            int x, int y, int s, int o,
                            const ImagePyramid<float>& gaussian_pyramid,
                            int num_scales)
  {
    const auto& G = gaussian_pyramid;

    // Fetch the following data.
    const auto& nearest_gaussian = G(s - 1, o);
    const auto gauss_truncate_factor = 4.f;
    const auto increase_sigma_max = sqrt(2.f);
    const auto patch_radius =
        int(ceil(increase_sigma_max * gauss_truncate_factor));

    // Ensure the patch is inside the image.
    if (x - patch_radius < 0 || x + patch_radius >= nearest_gaussian.width() ||
        y - patch_radius < 0 || y + patch_radius >= nearest_gaussian.height())
      return false;

    // First patch at the closest scale.
    auto nearest_patch = nearest_gaussian.compute<SafeCrop>(x, y, patch_radius);

    //#define DEBUG_SELECT_SCALE
#ifdef DEBUG_SELECT_SCALE
    // verbose.
#define print(variable) cout << #variable << " = " << variable << endl
    print_stage("Check patch variable");
    print(G.scale_relative_to_octave(s-1));
    print(G.scale_relative_to_octave(s));
    print(gauss_truncate_factor);
    print(increase_sigma_max);
    print(patch_radius);
    print(nearest_patch.sizes().transpose());

    // Debug
    auto zoom_factor = 10.;
    Window win = active_window() ? active_window() : 0;
    if (win)
      set_active_window(create_window(zoom_factor*nearest_patch.width(),
                                      zoom_factor*nearest_patch.height()) );
    display(nearest_patch, 0, 0, zoom_factor);
    get_key();
#endif

    // Store the blurred patches, their associated scales and LoG values at the
    // patch centers.
    auto patches = vector<Image<float>>(num_scales + 1);
    auto scales = vector<float>(num_scales + 1);
    auto LoGs = vector<float>(num_scales + 1);

    auto scale_common_ratio = pow(2.f, 1.f / num_scales);
    auto nearest_sigma = G.scale_relative_to_octave(s - 1);

#ifdef DEBUG_SELECT_SCALE
    print_stage("Print blur-related variables");
    print(scale_common_ratio);
    print(nearest_sigma);
#endif

    // Compute the blurred patches and their associated scales.
    //
    // Start with the initial patch.
    scales[0] = G.scale_relative_to_octave(s) / sqrt(2.f);
    auto inc_sigma = sqrt(pow(scales[0], 2) - pow(nearest_sigma, 2));
    patches[0] = inc_sigma > 1e-3f ?
      gaussian(nearest_patch, inc_sigma) :
      nearest_patch;

#ifdef DEBUG_SELECT_SCALE
    print_stage("Print sigma of each patch");
    print(scales[0]);
    print(inc_sigma);
    display(patches[0], 0, 0, zoom_factor);
    get_key();
#endif

    // Loop for the rest of the patches.
    for (size_t i = 1; i < patches.size(); ++i)
    {
      scales[i] = scale_common_ratio*scales[i-1];
      inc_sigma = sqrt(pow(scales[i], 2) - pow(scales[i - 1], 2));
      patches[i] = gaussian(patches[i - 1], inc_sigma);
#ifdef DEBUG_SELECT_SCALE
      print(scales[i]);
      print(inc_sigma);
      display(patches[i], 0, 0, zoom_factor);
      get_key();
#endif
    }

    // Compute the scale normalized LoG values in each patch centers
    for (size_t i = 0; i != patches.size(); ++i)
      LoGs[i] = laplacian(patches[i], Point2i(patch_radius,patch_radius))
              * pow(scales[i], 2);

    // Search local extremum.
    auto is_extremum = false;
    auto i = 1;
    for ( ; i < num_scales; ++i)
    {
      // Is LoG(\mathbf{x},\sigma) an extremum
      is_extremum = (LoGs[i] <= LoGs[i - 1] && LoGs[i] <= LoGs[i + 1]) ||
                    (LoGs[i] >= LoGs[i - 1] && LoGs[i] >= LoGs[i + 1]);
      if (is_extremum)
        break;
    }

    // Refine the extremum.
    if (is_extremum)
    {
      // Denote by $f$ be the LoG function, i.e.,
      // $\sigma \mapsto \sigma^2 (\Delta^2 I_\sigma)(\mathbf{x})$.
      // Use a 2nd-order Taylor approximation:
      // $f(x+h) = f(x) + f'(x)h + f''(x) h^2/2$
      // We approximate $f'$ and $f''$ by finite difference.
      auto fprime = (LoGs[i + 1] - LoGs[i - 1]) / 2.f;
      auto fsecond = LoGs[i - 1] - 2.f*LoGs[i] + LoGs[i + 1];
      // Maximize w.r.t. to $h$, derive the expression.
      // Thus $h = -f'(x)/f''(x)$.
      auto h = -fprime / fsecond;

      // OK, now the scale is:
      scale = scales[i] * pow(scale_common_ratio, h);
    }

#ifdef DEBUG_SELECT_SCALE
    closeWindow();
    if (win)
      set_active_window(win);
#endif

    return is_extremum;
  }

  vector<OERegion> laplace_maxima(const ImagePyramid<float>& function,
                                  const ImagePyramid<float>& gauss_pyramid,
                                  int s, int o,
                                  float extremum_thres,
                                  int img_padding_sz,
                                  int num_scales,
                                  int refine_iterations)
  {
    LocalMax<float> local_max;

    auto corners = vector<OERegion>{};
    corners.reserve(int(1e4));

    for (auto y = img_padding_sz; y < function(s,o).height()-img_padding_sz; ++y)
    {
      for (int x = img_padding_sz; x < function(s,o).width()-img_padding_sz; ++x)
      {
        if ( !local_max(x,y,function(s,o)) )
          continue;
        if ( function(x,y,s,o) < extremum_thres )
          continue;

        // Select the optimal scale using the normalized LoG.
        auto scale = function.scale_relative_to_octave(s);

        if (!select_laplace_scale(scale, x, y, s, o, gauss_pyramid, num_scales))
          continue;

        // Refine the spatial coordinates.
        auto val = function(x, y, s, o);
        auto p = Point2f(x, y);

        /*if (!refineExtremum(function(s,o),x,y,1,p,val,img_padding_sz,refine_iterations))
          continue;*/

        refine_extremum(function(s, o), x, y, 1, p, val, img_padding_sz,
                        refine_iterations);

        // Store the extremum.
        auto c = OERegion{};
        c.center() = p;
        c.shape_matrix() = Matrix2f::Identity()*pow(scale,-2);
        c.orientation() = 0.f;
        c.extremum_type() = OERegion::ExtremumType::Max;
        c.extremum_value() = val;
        corners.push_back(c);
      }
    }

    return corners;
  }

} /* namespace Sara */
} /* namespace DO */
