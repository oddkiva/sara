// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/ImageProcessing.hpp>
#include <DO/Graphics.hpp>

namespace DO {

  using namespace std;

  bool isNotEdge(Image<float>::ConstLocator& dog_loc, float k)
  {
    // Compute the Hessian matrix.
    float H00 = dog_loc.x()[1] - 2*(*dog_loc) + dog_loc.x()[-1];
    float H11 = dog_loc.y()[1] - 2*(*dog_loc) + dog_loc.y()[-1];
    float H01 = ((dog_loc(1,1) - dog_loc(1,-1)) 
               - (dog_loc(-1,1) - dog_loc(-1,-1))) / 4.f;

    // Based on Harris corner criterion.
    float det = H00*H11 - H01*H01;
    float tr = H00+H11;

    return det-k*tr > 0.f;
  }

  void ComputeScaleSpace::operator()(vector<Image<float> >& gaussians,
                                     vector<ScaleInfo>& scales,
                                     const Image<float>& image)
  {
    // Check scale-space parameters.
    cout << "Constructing scale space" << endl;
    checkParams();

    // Make a working copy of the image with the right dimensions
    Image<float> work;
    if (first_octave_ == - 1)
      work = enlarge(image, 2);
    else
      work = image;
    
    // Apply initial smoothing.
    float sigma = 0.5f * pow(2.f, first_octave_);
    float priorSmoothingFactor = 
      std::sqrt(initial_sigma_*initial_sigma_ - sigma*sigma);
    cout << "priorSmoothingFactor = " << priorSmoothingFactor << endl;
    //inPlaceDericheBlur(work, priorSmoothingFactor);
    applyGaussianFilter(work, work, priorSmoothingFactor);
    //view(work);

    // Fetch parameters.
    const float numScalesPerOct = 
      static_cast<float>(num_scales_per_octave_);
    const float sigmaRatio = std::pow(2.f, 1.f/numScalesPerOct);
    const float sigmaFactor = std::sqrt(sigmaRatio*sigmaRatio - 1.f);

    // Estimate the maximum number of octaves.
    const float w = image.width();
    const float h = image.height();
    const float b = border_size_;
    // We want 2*b < w / 2^k for all k <= kmax
    // where kmax == numOctaveMax
    // Thus, the following formula:
    const int numOctavesMax = static_cast<int>( std::min(
            std::log(w / (2.f*b)) / std::log(2.f),
            std::log(h / (2.f*b)) / std::log(2.f) ));
    cout << "numOctavesMax=" << numOctavesMax << endl;

    // Reserve the right number of images.
    if (!gaussians.empty())
      gaussians.clear();
    gaussians.reserve(numOctavesMax);

    // Compute the scale space.
    float octSize = (first_octave_ == -1) ? 0.5f : 1.0f;
    for (int octave = first_octave_; octave < numOctavesMax; ++octave)
    {
      //cout << "Octave #" << curOctave << endl;

      // Update octave size.
      sigma = initial_sigma_;
      
      // Smooth each scale of the octave.
      for (int scale = 0; scale < numScalesPerOct+3; ++scale)
      {
        //cout << "Scale #" << scale << endl;

        // Compute the incremental sigma factor.
        // Recall the formula:
        // g_{\sigma} * g_{\sigma'} = g_{\sqrt(\sigma^2 + \sigma'^2))}
        // Thus:
        // k^2 sigma^2 = sigma^2 + incSigma^2
        // Thus, the following formula:
        float incSigma = sigma*sigmaFactor;
        //inPlaceDericheBlur(work, incSigma);
        applyGaussianFilter(work, work, incSigma);

        // Update the previous sigma
        sigma *= sigmaRatio;

        gaussians.push_back(work);
        scales.push_back(ScaleInfo(sigma, octave, octSize));

        // View the obtained blurred image.
        //view(gaussians.back());
      }

      // Down-sample the image.
      work = scaleDown(work, 2);
      
      // Update octSize
      octSize /= 2.0f;
    }
  }

  void ComputeScaleSpace::checkParams() const
  {
    cout << "Scale space parameters" << endl;
    cout << "First octave: " << first_octave_ << endl;
    cout << "Num scales per octave: " << num_scales_per_octave_ << endl;
    cout << "Initial sigma: " << num_scales_per_octave_ << endl;
    cout << "Border size: " << border_size_ << endl;
  }

  void computeDoGs(vector<Image<float> >& dogs,
                   const vector<Image<float> >& gaussians)
  {
    typedef Image<float>::MatrixView MatView;
    typedef Image<float>::ConstMatrixView ConstMatView;

    if (gaussians.size() < 2)
    {
      cerr << "Wrong size of scale-space" << endl;
      return;
    }
    if (!dogs.empty())
      dogs.clear();
    dogs.reserve(gaussians.size());

    cout << "scaleSpace.size() = " << gaussians.size() << endl;

    for (size_t scaleIdx = 0; scaleIdx < gaussians.size()-1; ++scaleIdx)
    {
      if (gaussians[scaleIdx].width() != gaussians[scaleIdx+1].width())
        continue;

      dogs.push_back(Image<float>(gaussians[scaleIdx].sizes()));
      MatView dog = dogs.back().matrix();
      ConstMatView gp = gaussians[scaleIdx+1].matrix();
      ConstMatView gm = gaussians[scaleIdx].matrix();
      dog = gp - gm;
      //view(dogs.back());
    }

  }

  void computeDoHs(vector<Image<Matrix2f> >& dohs,
                   const vector<Image<float> >& gaussians)
  {
    typedef Image<Matrix2f> MatrixField;
    typedef Image<Matrix2f>::MatrixView MatView;

    if (gaussians.size() < 2)
    {
      cerr << "Wrong size of scale-space" << endl;
      return;
    }
    if (!dohs.empty())
      dohs.clear();
    dohs.reserve(gaussians.size());

    cout << "scaleSpace.size() = " << gaussians.size() << endl;

    vector<Image<Matrix2f> > hessians(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i)
    {
      ComputeHessian<float> computeHessian(gaussians[i]);
      hessians.push_back(computeHessian());
    }


    for (size_t i = 0; i < gaussians.size()-1; ++i)
    {
      //cout << "i = " << i << endl;
      if (gaussians[i].width() != gaussians[i+1].width())
        continue;

      dohs.push_back(MatrixField(gaussians[i].sizes()));
      MatView doh = hessians.back().matrix();
      MatView hp = hessians[i+1].matrix();
      MatView hm = hessians[i].matrix();
      doh = hp - hm;
    }

  }

} /* namespace DO */
