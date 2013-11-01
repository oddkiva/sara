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

#include <DO/FeatureDetectors.hpp>

using namespace std;

namespace DO {

  bool onEdge(const Image<float>& I, int x, int y, float edgeRatio)
  {
    Matrix2f H( hessian(I, Point2i(x,y)) );
    return pow(H.trace(), 2)*edgeRatio >= 
           pow(edgeRatio+1.f, 2)*fabs(H.determinant());
  }

  bool refineExtremum(const ImagePyramid<float>& I, int x, int y, int s, int o,
                      int type, Point3f& pos, float& val, int borderSz, 
                      int numIter)
  {
    Vector3f Dprime; // gradient
    Matrix3f Dsecond; // hessian
    Vector3f h;
    Vector3f lambda;
    
    pos = Vector3f(float(x),float(y),I.octRelScale(s));

    int i;
    for (i = 0; i < numIter; ++i)
    {
      // Range check at each iteration. The first iteration should always be OK.
      if (x < borderSz || x >= I(s,o).width()-borderSz  ||
          y < borderSz || y >= I(s,o).height()-borderSz ||
          s < 1 || s >= static_cast<int>(I(o).size())-1)
        break;

      // Estimate the gradient and the hessian matrix by central finite 
      // differentiation.
      Dprime = gradient(I,x,y,s,o);
      Dsecond = hessian(I,x,y,s,o);

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
      SelfAdjointEigenSolver<Matrix3f> solver(Dsecond);
      lambda = solver.eigenvalues();
      // Not sure about numerical errors... But it might just work out for now.
      if ((lambda*float(type)).maxCoeff() >= 0)
      {
        h.setZero();
        break;
      }

      // $D''(\mathbf{x})$ is just a 3x3 matrix and computing its inverse is 
      // thus cheap (cf. Eigen library.).
      h = -Dsecond.inverse()*Dprime;

      // The interpolated extremum should be normally close to the initial 
      // position which has integral coordinates. Otherwise, the estimates of 
      // the gradient and the Hessian matrix are bad.
      if (h.block(0,0,2,1).cwiseAbs().maxCoeff() > 1.5f)
      {
        //#define VERBOSE
  #ifdef VERBOSE
        printStage("Offset is too large: don't refine");
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

    pos = Vector3f(float(x),float(y),I.octRelScale(s));
    float oldval = I(x,y,s,o);
    float newval = oldval + 0.5f*Dprime.dot(h);

    if ( (type==1 && oldval <= newval) || (type==-1 && oldval >= newval) )
    {
      pos += h;
      val = newval;
    }
    else
      std::cerr << "INTERPOLATION ERROR" << std::endl;

    return true;
  }

  bool refineExtremum(const Image<float>& I, int x, int y, int type,
                      Point2f& pos, float& val, int borderSz, int numIter)
  {
    Vector2f Dprime; // gradient
    Matrix2f Dsecond; // hessian
    Vector2f h; // offset to estimate

    int i;
    for (i = 0; i < numIter; ++i)
    {
      // Range check at each iteration. The first iteration should always be OK.
      if (x < borderSz || x >= I.width()-borderSz  ||
          y < borderSz || y >= I.height()-borderSz )
        break;
      
      // Estimate the gradient and the hessian matrix by central finite 
      // differentiation.
      Dprime = gradient(I,Point2i(x,y));
      Dsecond = hessian(I,Point2i(x,y));
      
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
      if (Dsecond.determinant() <= 0.f || Dsecond.trace()*type >= 0.f)
      {
        Dprime.setZero();
        break;
      }
      
      // $D''(\mathbf{x})$ is just a 3x3 matrix and computing its inverse is 
      // thus cheap (cf. Eigen library.).
      h = -Dsecond.inverse()*Dprime;
      
      // The interpolated extremum should be normally close to the initial 
      // position which has integral coordinates. Otherwise, the estimates of 
      // the gradient and the Hessian matrix are bad.
      if (h.cwiseAbs().maxCoeff() > 1.5f)
      {
        //#define VERBOSE
#ifdef VERBOSE
        printStage("Offset is too large: don't refine");
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

    pos << x, y;
    float oldval = I(x,y);
    float newval = oldval + 0.5f*Dprime.dot(h);

    if ( (type==1 && oldval <= newval) || (type==-1 && oldval >= newval) )
    {
      pos += h;
      val = newval;
    }
    else
      std::cerr << "INTERPOLATION ERROR" << std::endl;

    return true;
  }

  vector<OERegion> localScaleSpaceExtrema(const ImagePyramid<float>& I,
                                          int s, int o, 
                                          float extremumThres,
                                          float edgeRatioThres,
                                          int imgPaddingSz,
                                          int refineIter)
  {
    std::vector<OERegion> extrema;
    extrema.reserve(10000);

    Image<int> map(I(s,o).sizes());
    map.array().setZero();

//#define STRICT_LOCAL_EXTREMA
#ifdef STRICT_LOCAL_EXTREMA
    LocalScaleSpaceExtremum<std::greater, float> local_max;
    LocalScaleSpaceExtremum<std::less, float> local_min;
#else
    LocalScaleSpaceExtremum<std::greater_equal, float> local_max;
    LocalScaleSpaceExtremum<std::less_equal, float> local_min;
#endif

    for (int y = imgPaddingSz; y < I(s,o).height()-imgPaddingSz; ++y)
    {
      for (int x = imgPaddingSz; x < I(s,o).width()-imgPaddingSz; ++x)
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
        if (std::abs(I(x,y,s,o)) < 0.8f*extremumThres)
          continue;
#endif
        // Reject early if located on edge.
        if (onEdge(I(s,o),x,y,edgeRatioThres))
          continue;
        // Try to refine extremum.
        Point3f pos;
        float val;
        /*if (!refineExtremum(I,x,y,s,o,type,pos,val,imgPaddingSz,refineIter))
          continue;*/
        refineExtremum(I,x,y,s,o,type,pos,val,imgPaddingSz,refineIter);
        
        // Don't add if already marked.
        if (map(static_cast<int>(x), static_cast<int>(y)) == 1)
          continue;
#ifndef STRICT_LOCAL_EXTREMA
        // Reject if contrast too low.
        if (std::abs(val) < extremumThres)
          continue;
#endif
        // Store the DoG extremum.
        OERegion dog(pos.head<2>(), pos.z());

        dog.extremumValue() = val;
        dog.extremumType() = type == 1 ? OERegion::Max : OERegion::Min;
        extrema.push_back(dog);
        map(static_cast<int>(x), static_cast<int>(y)) = 1;
      }
    }

    return extrema;
  }

  
  bool selectLaplaceScale(float& scale,
                          int x, int y, int s, int o,
                          const ImagePyramid<float>& gaussPyramid,
                          int numScales)
  {
    const ImagePyramid<float>& G = gaussPyramid;

    // Fetch the following data.
    const Image<float>& nearestGaussian = G(s-1,o);
    float gaussTruncFactor = 4.f;
    float incSigmaMax = sqrt(2.f);
    int patchRadius = int(ceil(incSigmaMax*gaussTruncFactor)); // patch radius

    // Ensure the patch is inside the image.
    if ( x-patchRadius < 0 || x+patchRadius >= nearestGaussian.width() ||
         y-patchRadius < 0 || y+patchRadius >= nearestGaussian.height() )
      return false;
  
    // First patch at the closest scale.
    Image<float> nearestPatch( getImagePatch(nearestGaussian,x,y,patchRadius) );
//#define DEBUG_SELECT_SCALE
#ifdef DEBUG_SELECT_SCALE
    // verbose.
#define print(variable) cout << #variable << " = " << variable << endl
    printStage("Check patch variable");
    print(G.octRelScale(s-1));
    print(G.octRelScale(s));
    print(gaussTruncFactor);
    print(incSigmaMax);
    print(patchRadius);
    print(nearestPatch.sizes().transpose());
    // Debug
    double zoomFactor = 10.;
    Window win = activeWindow() ? activeWindow() : 0;
    if (win)
      setActiveWindow(openWindow(zoomFactor*nearestPatch.width(),
      zoomFactor*nearestPatch.height()) );
    display(nearestPatch, 0, 0, zoomFactor);
    getKey();
#endif
    // Store the blurred patches, their associated scales and LoG values at the
    // patch centers.
    vector<Image<float> > patches(numScales+1);
    vector<float> scales(numScales+1);
    vector<float> LoGs(numScales+1);
  
    float scaleCommonRatio = pow(2.f, 1.f/numScales);
    float nearestSigma = G.octRelScale(s-1);
#ifdef DEBUG_SELECT_SCALE
    printStage("Print blur-related variables");
    print(scaleCommonRatio);
    print(nearestSigma);
#endif
    // Compute the blurred patches and their associated scales.
    //
    // Start with the initial patch.
    scales[0] = G.octRelScale(s)/sqrt(2.f);
    float incSigma = sqrt(pow(scales[0], 2) - pow(nearestSigma, 2));
    patches[0] = incSigma > 1e-3f ? 
      gaussian(nearestPatch, incSigma) :
      nearestPatch;
#ifdef DEBUG_SELECT_SCALE
    printStage("Print sigma of each patch");
    print(scales[0]);
    print(incSigma);
    display(patches[0], 0, 0, zoomFactor);
    getKey();
#endif
    // Loop for the rest of the patches.
    for (size_t i = 1; i < patches.size(); ++i)
    {
      scales[i] = scaleCommonRatio*scales[i-1];
      incSigma = sqrt(pow(scales[i],2) - pow(scales[i-1], 2));
      patches[i] = gaussian(patches[i-1], incSigma);
#ifdef DEBUG_SELECT_SCALE
      print(scales[i]);
      print(incSigma);
      display(patches[i], 0, 0, zoomFactor);
      getKey();
#endif
    }
    // Compute the scale normalized LoG values in each patch centers
    for (size_t i = 0; i != patches.size(); ++i)
      LoGs[i] = laplacian(patches[i], Point2i(patchRadius,patchRadius))
              * pow(scales[i], 2);

    // Search local extremum
    bool isExtremum = false;
    int i = 1;
    for ( ; i < numScales; ++i)
    {
      // Is LoG(\mathbf{x},\sigma) an extremum                
      isExtremum = (LoGs[i] <= LoGs[i-1] && LoGs[i] <= LoGs[i+1]) || 
                   (LoGs[i] >= LoGs[i-1] && LoGs[i] >= LoGs[i+1]) ;
      if (isExtremum)
        break;
    }
 
    // Refine the extremum.
    if (isExtremum)
    {
      // Denote by $f$ be the LoG function, i.e.,
      // $\sigma \mapsto \sigma^2 (\Delta^2 I_\sigma)(\mathbf{x})$.
      // Use a 2nd-order Taylor approximation:
      // $f(x+h) = f(x) + f'(x)h + f''(x) h^2/2$
      // We approximate $f'$ and $f''$ by finite difference.
      float fprime = (LoGs[i+1]-LoGs[i-1]) / 2.f;
      float fsecond = LoGs[i-1] - 2.f*LoGs[i] + LoGs[i+1];
      // Maximize w.r.t. to $h$, derive the expression.
      // Thus $h = -f'(x)/f''(x)$.
      float h = -fprime / fsecond;
      // OK, now the scale is:
      scale = scales[i]*pow(scaleCommonRatio, h);
    }
#ifdef DEBUG_SELECT_SCALE
    closeWindow();
    if (win)
      setActiveWindow(win);
#endif
    return isExtremum;
  }

  vector<OERegion> laplaceMaxima(const ImagePyramid<float>& function,
                                  const ImagePyramid<float>& gaussPyramid,
                                  int s, int o,
                                  float extremumThres,
                                  float imgPaddingSz,
                                  float numScales,
                                  int refineIter)
  {
    LocalMax<float> localMax;

    vector<OERegion> corners;
    corners.reserve(1e4);
    for (int y = imgPaddingSz; y < function(s,o).height()-imgPaddingSz; ++y)
    {
      for (int x = imgPaddingSz; x < function(s,o).width()-imgPaddingSz; ++x)
      {
        if ( !localMax(x,y,function(s,o)) )
          continue;
        if ( function(x,y,s,o) < extremumThres )
          continue;
        // Select the optimal scale using the normalized LoG.
        float scale = function.octRelScale(s);
        if (!selectLaplaceScale(scale,x,y,s,o,gaussPyramid,numScales))
          continue;
        // Refine the spatial coordinates.
        float val = function(x,y,s,o);
        Point2f p(x,y);
        if (!refineExtremum(function(s,o),x,y,1,p,val,imgPaddingSz,refineIter))
          continue;        
        // Store the extremum.
        OERegion c;
        c.center() = p;
        c.shapeMat() = Matrix2f::Identity()*pow(scale,-2);
        c.orientation() = 0.f;
        c.extremumType() = OERegion::Max;
        c.extremumValue() = val;
        corners.push_back(c);
      }
    }
    return corners;
  }

} /* namespace DO */