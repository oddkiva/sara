/*
    Compute the affinity that maps the normalized patch to the local region
    around feature $f$.
    We denote a point in the normalized patch by $(u,v) \in [0,w]^2$
    The center point is $(w/2, w/2)$ corresponds to the center $(x_f, y_f)$
    of feature $f$.

    We introduce the notion of 'scale unit', i.e.,
    $1$ scale unit is equivalent $\sigma$ pixels in the image.
  */
  /* 
    Let us set some important constants needed for the computation of the 
    normalized patch computation. 
   */
  // Patch "radius"
  const int patchRadius = 20;
  // Patch side length
  const int patchSideLength = 2*patchRadius+1;
  // Gaussian smoothing is involved in the computation of gradients orientations
  // to compute dominant orientations and the SIFT descriptor.
  const float gaussTruncFactor = 3.f; 
  // A normalized patch is composed of a grid of NxN square patches, i.e. bins,
  // centered on the feature
  const float binSideLength = 3.f; // side length of a bin in scale unit.
  const float numBins = 4.f;
  const float scaleRelRadius = sqrt(2.f)*binSideLength*(numBins+1)/2.f;
  // Store the keypoints here.
  vector<OERegion> keptFeats;
  keptFeats.reserve(2*feats.size());
  for (size_t i = 0; i != feats.size(); ++i)
  {
    if (keepFeatures[i] == 1)
    {
      // The linear transform computed from the SVD
      const Matrix2f& shapeMat = feats[i].shapeMat();
      JacobiSVD<Matrix2f> svd(shapeMat, ComputeFullU);
      Vector2f S(svd.singularValues().cwiseInverse().cwiseSqrt());
      S *= scaleRelRadius/patchRadius; // Scaling
      Matrix2f U(svd.matrixU()); // Rotation
      Matrix2f L(U*S.asDiagonal()*U.transpose()); // Linear transform.
      // The translation vector
      Vector2f t(L*Point2f::Ones()*(-patchRadius) + feats[i].center());
      // The affinity that maps the patch to the local region around the feature
      Matrix3f T(Matrix3f::Zero());
      T.block<2,2>(0,0) = L;
      T.col(2) << t, 1.f;

      // Get the normalized patch.
      Image<float> normalizedPatch(patchSideLength,patchSideLength);
      int s = scaleOctPairs[i](0);
      int o = scaleOctPairs[i](1);
      if (!warp(normalizedPatch, gaussPyr(s,o), T, 0.f, true))
        continue;


      // Rescale the feature position and shape to the original image
      // dimensions.
      double fact = gaussPyr.octaveScalingFactor(o);
      feats[i].shapeMat() *= pow(fact/**scaleRelRadius*/, -2);
      feats[i].center() *= fact;
      // Store the keypoint.
      keptFeats.push_back(feats[i]);
    }
  }
  if (verbose)
    toc();