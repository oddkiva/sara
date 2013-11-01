#include <DO/FeatureDetectors.hpp>
#include <DO/FeatureDescriptors.hpp>
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

Set<OERegion, RealDescriptor> computeSIFT(const Image<float>& image)
{
  // Time everything.
  HighResTimer timer;
  double elapsed = 0.;
  double DoGDetTime, oriAssignTime, siftDescTime, gradGaussianTime;

  // We describe the work flow of the feature detection and description.
  Set<OERegion, RealDescriptor> keys;
  vector<OERegion>& DoGs = keys.features;
  DescriptorMatrix<float>& SIFTDescriptors = keys.descriptors;

  // 1. Feature extraction.
  printStage("Computing DoG extrema");
  timer.restart();
  ImagePyramidParams pyrParams(0);
  ComputeDoGExtrema computeDoGs(pyrParams, 0.005f);
  vector<Point2i> scaleOctPairs;
  DoGs = computeDoGs(image, &scaleOctPairs);
  DoGDetTime = timer.elapsedMs();
  elapsed += DoGDetTime;
  cout << "DoG detection time = " << DoGDetTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  printStage("Computing gradients of Gaussians");
  timer.restart();
  ImagePyramid<Vector2f> gradG;
  gradG = gradPolar(computeDoGs.gaussians());
  gradGaussianTime = timer.elapsedMs();
  elapsed += gradGaussianTime;
  cout << "gradient of Gaussian computation time = " << gradGaussianTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // Find dominant gradient orientations.
  printStage("Assigning (possibly multiple) dominant orientations to DoG extrema");
  timer.restart();
  ComputeDominantOrientations assignOrientations;
  assignOrientations(gradG, DoGs, scaleOctPairs);
  oriAssignTime = timer.elapsedMs();
  elapsed += oriAssignTime;
  cout << "orientation assignment time = " << oriAssignTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // 3. Feature description.
  printStage("Describe DoG extrema with SIFT descriptors");
  timer.restart();
  ComputeSIFTDescriptor<> computeSIFT;
  SIFTDescriptors = computeSIFT(DoGs, scaleOctPairs, gradG);
  siftDescTime = timer.elapsedMs();
  elapsed += siftDescTime;
  cout << "description time = " << siftDescTime << " ms" << endl;
  cout << "sifts.size() = " << SIFTDescriptors.size() << endl;

  // Summary in terms of computation time.
  printStage("Total Detection/Description time");
  cout << "SIFT computation time = " << elapsed << " ms" << endl;

  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    float octScaleFact = gradG.octaveScalingFactor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shapeMat() /= pow(octScaleFact, 2);
  }

  return keys;
}

bool checkDescriptors(const DescriptorMatrix<float>& descriptors)
{
  for (int i = 0; i < descriptors.size(); ++i)
  {
    for (int j = 0; j < descriptors.dimension(); ++j)
    {
      if (!DO::isfinite(descriptors[i](j)))
      {
        cerr << "Not a finite number" << endl;
        return false;
      }
    }
  }
  cout << "OK all numbers are finite" << endl;
  return true;
}

int main()
{
  Image<float> image;
  if (!load(image, srcPath("sunflowerField.jpg")))
    return -1;

  printStage("Detecting SIFT features");
  Set<OERegion, RealDescriptor> SIFTs = computeSIFT(image.convert<float>());
  const vector<OERegion>& features = SIFTs.features; 

  printStage("Removing existing redundancies");
  removeRedundancies(SIFTs);
  CHECK(SIFTs.features.size());
  CHECK(SIFTs.descriptors.size());

  // Check the features visually.
  printStage("Draw features");
  openWindow(image.width(), image.height());
  setAntialiasing();
  display(image);
  for (size_t i=0; i != features.size(); ++i)
    features[i].draw(features[i].extremumType() == OERegion::Max ? Red8 : Blue8);
  getKey();
  

  return 0;
}