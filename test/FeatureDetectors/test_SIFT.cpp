#include <DO/FeatureDetectors.hpp>
#include <DO/Graphics.hpp>

using namespace DO;
using namespace std;

#ifdef CHECK_STEP_BY_STEP
void testDoGSIFTKeypoints(const Image<float>& I)
{
  Window imageWin = openWindow(I.width(), I.height());
  setAntialiasing();

  ImagePyramid<float> G(gaussianPyramid(I));
  //checkImagePyramid(G);

  ImagePyramid<float> D(DoGPyramid(G));
  //checkImagePyramid(D, true);

  for (int o = 0; o < D.numOctaves(); ++o)
  {
    // Verbose.
    printStage("Processing octave");
    cout << "Octave " << o << endl;
    cout << "Octave scaling factor = " << D.octaveScalingFactor(o) << endl;

    // Be careful of the bounds. We go from 1 to N-1.
    for (int s = 1; s < D.numScalesPerOctave()-1; ++s)
    {
      vector<OERegion> extrema( localScaleSpaceExtrema(D,s,o) );

      // Verbose.
      printStage("Detected extrema");
      cout << "[" << s << "] sigma = " << D.scale(s,o) << endl;
      cout << "    num extrema = " << extrema.size() << endl;

      // Draw the keypoints.
      //display(I.convert<float>());
      drawExtrema(D, extrema, s, o);
      getKey();

      // Gradient in polar coordinates.
      Image<Vector2f> gradG( gradPolar(G(s,o)) );

      // Determine orientations.
      drawExtrema(G, extrema, s, o, false);
      for (size_t i = 0; i != extrema.size(); ++i)
      {
#define DEBUG_ORI
#ifdef DEBUG_ORI
        // Draw the patch on the image.
        highlightPatch(D, extrema[i].x(), extrema[i].y(), extrema[i].scale(), o);

        // Close-up on the image patch
        checkPatch(G(s,o), extrema[i].x(), extrema[i].y(), extrema[i].scale());

        // Orientation histogram
        printStage("Orientation histogram");
#endif
        Array<float, 36, 1> oriHist;
        computeOrientationHistogram(
          oriHist, gradG,
          extrema[i].x(),
          extrema[i].y(),
          extrema[i].scale());
        viewHistogram(oriHist);
        // Note that the peaks are shifted after smoothing.
#ifdef DEBUG_ORI
        printStage("Smoothing orientation histogram");
#endif
        smoothHistogram_Lowe(oriHist);
        viewHistogram(oriHist);
        // Orientation peaks.
#ifdef DEBUG_ORI
        printStage("Localizing orientation peaks");
#endif
        vector<int> oriPeaks(findPeaks(oriHist));
#ifdef DEBUG_ORI
        cout << "Raw peaks" << endl;
        for (size_t k = 0; k != oriPeaks.size(); ++k)
          cout << oriPeaks[k]*10 << endl;

        // Refine peaks.
        printStage("Refining peaks");
#endif
        vector<float> refinedOriPeaks(refinePeaks(oriHist, oriPeaks) );
#ifdef DEBUG_ORI
        cout << "Refined peaks" << endl;
        for (size_t k = 0; k != refinedOriPeaks.size(); ++k)
          cout << refinedOriPeaks[k]*10 << endl;
#endif
        // Convert orientation to radian.
        Map<ArrayXf> peaks(&refinedOriPeaks[0], refinedOriPeaks.size());
        peaks *= float(M_PI)/36;

        setActiveWindow(imageWin);
        printStage("Compute SIFT descriptor");
        for (int ori = 0; ori < refinedOriPeaks.size(); ++ori)
        {
          ComputeSIFTDescriptor<> computeSift;

          printStage("Draw patch grid");
          computeSift.drawGrid(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), D.octaveScalingFactor(o), 3);
          getKey();

          printStage("Compute SIFT descriptor with specified orientation");
          Matrix<unsigned char, 128, 1> sift;
          sift = computeSift(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), gradG);
        }
      }
      getKey();
    }
  }

  closeWindow();
}
#endif

int main()
{
  Image<Rgb8> I;
  if (!load(I, srcPath("sunflowerField.jpg")))
    return -1;
  //I = enlarge(I,4);

  // We describe the work flow of the feature detection and description.

  // 1. Feature extraction.
  printStage("Computing DoG extrema");
  ImagePyramidParams pyrParams(-1);
  ComputeDoGExtrema computeDoGs(pyrParams);
  vector<OERegion> dogs;
  vector<Point2i> scaleOctPairs;
  dogs = computeDoGs(I.convert<float>(), &scaleOctPairs);
  cout << "DoGs.size() = " << dogs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  printStage("Computing gradients of Gaussians");
  ImagePyramid<Vector2f> gradG;
  gradG = gradPolar(computeDoGs.gaussians());
  // Find dominant gradient orientations.
  printStage("Assigning (possibly multiple) dominant orientations to DoG extrema");
  ComputeDominantOrientations assignOri;
  assignOri(gradG, dogs, scaleOctPairs);
  cout << "DoGs.size() = " << dogs.size() << endl;

  // 3. Feature description.
  printStage("Describe DoG extrema with SIFT descriptors");
  ComputeSIFTDescriptor<> computeSIFT;
  std::vector<ComputeSIFTDescriptor<>::SIFT> sifts;
  sifts = computeSIFT(dogs, scaleOctPairs, gradG);
  cout << "sifts.size() = " << sifts.size() << endl;

  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  printStage("Draw features");
  for (size_t i = 0; i != dogs.size(); ++i)
  {
    float octScaleFact = gradG.octaveScalingFactor(scaleOctPairs[i](1));
    dogs[i].center() *= octScaleFact;
    dogs[i].shapeMat() /= pow(octScaleFact, 2);
  }

  // 5. Check the features visually.
  openWindow(I.width(), I.height());
  display(I.convert<float>());
  setAntialiasing();
  for (size_t i=0; i != dogs.size(); ++i)
  {
    /*cout << dogs[i].center().transpose() << " " 
         << dogs[i].scale() << " "
         << dogs[i].orientation() << endl;*/
    dogs[i].drawOnScreen(dogs[i].extremumType() == OERegion::Max ? Red8 : Blue8);
    //getKey();
  }
  getKey();

  return 0;
}