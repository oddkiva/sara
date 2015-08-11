#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/Graphics.hpp>

using namespace DO;
using namespace std;

#ifdef CHECK_STEP_BY_STEP
void testDoGSIFTKeypoints(const Image<float>& I)
{
  Window imageWin = create_window(I.width(), I.height());
  set_antialiasing();

  ImagePyramid<float> G(gaussianPyramid(I));
  //checkImagePyramid(G);

  ImagePyramid<float> D(DoGPyramid(G));
  //checkImagePyramid(D, true);

  for (int o = 0; o < D.numOctaves(); ++o)
  {
    // Verbose.
    print_stage("Processing octave");
    cout << "Octave " << o << endl;
    cout << "Octave scaling factor = " << D.octave_scaling_factor(o) << endl;

    // Be careful of the bounds. We go from 1 to N-1.
    for (int s = 1; s < D.numScalesPerOctave()-1; ++s)
    {
      vector<OERegion> extrema( localScaleSpaceExtrema(D,s,o) );

      // Verbose.
      print_stage("Detected extrema");
      cout << "[" << s << "] sigma = " << D.scale(s,o) << endl;
      cout << "    num extrema = " << extrema.size() << endl;

      // Draw the keypoints.
      //display(I.convert<float>());
      drawExtrema(D, extrema, s, o);
      get_key();

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
        print_stage("Orientation histogram");
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
        print_stage("Smoothing orientation histogram");
#endif
        smoothHistogram_Lowe(oriHist);
        viewHistogram(oriHist);
        // Orientation peaks.
#ifdef DEBUG_ORI
        print_stage("Localizing orientation peaks");
#endif
        vector<int> oriPeaks(findPeaks(oriHist));
#ifdef DEBUG_ORI
        cout << "Raw peaks" << endl;
        for (size_t k = 0; k != oriPeaks.size(); ++k)
          cout << oriPeaks[k]*10 << endl;

        // Refine peaks.
        print_stage("Refining peaks");
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
        print_stage("Compute SIFT descriptor");
        for (int ori = 0; ori < refinedOriPeaks.size(); ++ori)
        {
          ComputeSIFTDescriptor<> computeSift;

          print_stage("Draw patch grid");
          computeSift.drawGrid(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), D.octave_scaling_factor(o), 3);
          get_key();

          print_stage("Compute SIFT descriptor with specified orientation");
          Matrix<unsigned char, 128, 1> sift;
          sift = computeSift(
            extrema[i].x(), extrema[i].y(), extrema[i].scale(),
            peaks(ori), gradG);
        }
      }
      get_key();
    }
  }

  closeWindow();
}
#endif

Set<OERegion, RealDescriptor> computeSIFT(const Image<float>& image)
{
  // Time everything.
  Timer timer;
  double elapsed = 0.;
  double DoGDetTime, oriAssignTime, siftDescTime, gradGaussianTime;

  // We describe the work flow of the feature detection and description.
  Set<OERegion, RealDescriptor> keys;
  vector<OERegion>& DoGs = keys.features;
  DescriptorMatrix<float>& SIFTDescriptors = keys.descriptors;

  // 1. Feature extraction.
  print_stage("Computing DoG extrema");
  timer.restart();
  ImagePyramidParams pyrParams;//(0);
  ComputeDoGExtrema computeDoGs(pyrParams, 0.01f);
  vector<Point2i> scaleOctPairs;
  DoGs = computeDoGs(image, &scaleOctPairs);
  DoGDetTime = timer.elapsed_ms();
  elapsed += DoGDetTime;
  cout << "DoG detection time = " << DoGDetTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  print_stage("Computing gradients of Gaussians");
  timer.restart();
  ImagePyramid<Vector2f> gradG;
  gradG = gradPolar(computeDoGs.gaussians());
  gradGaussianTime = timer.elapsed_ms();
  elapsed += gradGaussianTime;
  cout << "gradient of Gaussian computation time = " << gradGaussianTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // Find dominant gradient orientations.
  print_stage("Assigning (possibly multiple) dominant orientations to DoG extrema");
  timer.restart();
  ComputeDominantOrientations assignOrientations;
  assignOrientations(gradG, DoGs, scaleOctPairs);
  oriAssignTime = timer.elapsed_ms();
  elapsed += oriAssignTime;
  cout << "orientation assignment time = " << oriAssignTime << " ms" << endl;
  cout << "DoGs.size() = " << DoGs.size() << endl;


  // 3. Feature description.
  print_stage("Describe DoG extrema with SIFT descriptors");
  timer.restart();
  ComputeSIFTDescriptor<> computeSIFT;
  SIFTDescriptors = computeSIFT(DoGs, scaleOctPairs, gradG);
  siftDescTime = timer.elapsed_ms();
  elapsed += siftDescTime;
  cout << "description time = " << siftDescTime << " ms" << endl;
  cout << "sifts.size() = " << SIFTDescriptors.size() << endl;

  // Summary in terms of computation time.
  print_stage("Total Detection/Description time");
  cout << "SIFT computation time = " << elapsed << " ms" << endl;

  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    float octScaleFact = gradG.octave_scaling_factor(scaleOctPairs[i](1));
    DoGs[i].center() *= octScaleFact;
    DoGs[i].shape_matrix() /= pow(octScaleFact, 2);
  }

  return keys;
}

bool check_descriptors(const DescriptorMatrix<float>& descriptors)
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
  if (!load(image, src_path("../../datasets/sunflowerField.jpg")))
    return -1;

  print_stage("Detecting SIFT features");
  Set<OERegion, RealDescriptor> SIFTs = computeSIFT(image.convert<float>());
  const vector<OERegion>& features = SIFTs.features;

  print_stage("Removing existing redundancies");
  removeRedundancies(SIFTs);
  CHECK(SIFTs.features.size());
  CHECK(SIFTs.descriptors.size());

  // Check the features visually.
  print_stage("Draw features");
  create_window(image.width(), image.height());
  set_antialiasing();
  display(image);
  for (size_t i=0; i != features.size(); ++i)
    features[i].draw(features[i].extremum_type() == OERegion::Max ? Red8 : Blue8);
  get_key();


  return 0;
}