#include <DO/Core.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>
#include <DO/Match.hpp>
#include <DO/Geometry.hpp>
#include <DO/FeatureDetectors.hpp>
#include <DO/FeatureDescriptors.hpp>
#include <DO/FeatureMatching.hpp>
#include "GrowMultipleRegions.hpp"

using namespace std;
using namespace DO;

// ========================================================================== //
// Helper functions.
inline Rgb8 randRgb8()
{ return Rgb8(rand()%256, rand()%256, rand()%256); }

template <typename T>
Image<T> rotateCCW(const Image<T>& image)
{
  Image<T> dst(image.height(), image.width());
  // Transpose.
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x)
      dst(y,x) = image(x,y);
  // Reverse rows.
  for (int y = 0; y < dst.height(); ++y)
    for (int x = 0; x < dst.width(); ++x)
    {
      int n_x = dst.width()-1-x;
      if (x >= n_x)
        break;
      std::swap(dst(x,y), dst(n_x,y));
    }
    return dst;
}

Window openImgPairWindow(const Image<Rgb8>& image1, const Image<Rgb8>& image2,
                         float scale)
{
  int w = int((image1.width()+image2.width())*scale);
  int h = int(max(image1.height(), image2.height())*scale);
  return openWindow(w, h);
}

void dilateKeyScales(vector<OERegion>& features, float dilationFactor)
{
  for (size_t i = 0; i != features.size(); ++i)
    features[i].shapeMat() /=  dilationFactor*dilationFactor;
}


// ========================================================================== //
// SIFT Detector.
class SIFTDetector
{
public:
  SIFTDetector()
    : firstOctave(-1)
    , numOctaves(-1)
    , numScales(3)
    , edgeThresh(10.0f)
    , peakThresh(0.04f)
  {
  }

  void setNumOctaves(int n)	{ numOctaves=n; }
  void setNumScales(int n)	{ numScales=n; }
  void setFirstOctave(int n)	{ firstOctave=n; }
  void setEdgeThresh(float t) { edgeThresh=t; }
  void setPeakThresh(float t) { peakThresh=t; }

  //! N.B.: I have multiplied the scale by 8. to have higher circle.
  //! Rationale behind this is that SIFT descriptor are computed over a 
  //! 16*16 window patch at the scale = 1.
  Set<OERegion, RealDescriptor> run(const Image<float>& image) const
  {
    Set<OERegion, RealDescriptor> keys;
    vector<OERegion>& DoGs = keys.features;
    DescriptorMatrix<float>& SIFTDescriptors = keys.descriptors;

    // Time everything.
    HighResTimer timer;
    double elapsed = 0.;
    double DoGDetectionTime, oriAssignTime, SIFTDescriptionTime, gradGaussianTime;

    // 1. Feature extraction.
    printStage("Computing DoG extrema");
    timer.restart();
    ImagePyramidParams pyrParams;//(0);
    ComputeDoGExtrema computeDoGs(pyrParams, 0.01f);
    vector<Point2i> scaleOctPairs;
    DoGs = computeDoGs(image, &scaleOctPairs);
    DoGDetectionTime = timer.elapsedMs();
    elapsed += DoGDetectionTime;
    cout << "DoG detection time = " << DoGDetectionTime << " ms" << endl;
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
    SIFTDescriptionTime = timer.elapsedMs();
    elapsed += SIFTDescriptionTime;
    cout << "description time = " << SIFTDescriptionTime << " ms" << endl;
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

private:
  // First Octave Index. 
  int firstOctave;
  // Number of octaves.
  int numOctaves;	
  // Number of scales per octave. 
  int numScales;  
  // Max ratio of Hessian eigenvalues. 
  float edgeThresh;
  // Min contrast.
  float peakThresh;
};


// ========================================================================== //
// Matching demo.
void imagePairMatching() 
{
  HighResTimer timer;
  double elapsed;

  // Where are the images?
  string shelfPath = srcPath("shelves/shelf-1.jpg");
  string productPath = srcPath("products/garnier-shampoing.jpg");

  // Load the shelf and product images.
  Image<Rgb8> shelf, product;
  if (!load(shelf, shelfPath))
  {
    cerr << "Cannot load shelf image: " << shelfPath << endl;
    return;
  }
  if (!load(product, productPath))
  {
    cerr << "Cannot load product image: " << productPath << endl;
    return;
  }
  shelf = rotateCCW(shelf);

  // Run GUI graphics.
  float scale = 0.5;
  Window imagePairWin = openImgPairWindow(product, shelf, scale);
  setAntialiasing();
  PairWiseDrawer drawer(product, shelf);
  drawer.setVizParams(scale, scale, PairWiseDrawer::CatH);
  drawer.displayImages();

  // Detect keys.
  printStage("Detecting SIFT keypoints");
  Set<OERegion, RealDescriptor> shelfKeys, productKeys;
  SIFTDetector detector;
  detector.setFirstOctave(0);
  timer.restart();
  shelfKeys = detector.run(shelf.convert<float>());
  productKeys = detector.run(product.convert<float>());
  elapsed = timer.elapsedMs();
  cout << "Detection time = " << elapsed << " ms" << endl;

  // Dilate SIFT scales before matching keypoints.
  const float scaleFactor = 4.f;
  dilateKeyScales(shelfKeys.features, scaleFactor);
  dilateKeyScales(productKeys.features, scaleFactor);

  // Compute initial matches.
  printStage("Compute initial matches");
  float lowe_ratio_threshold = 1.f;
  AnnMatcher matcher(productKeys, shelfKeys, lowe_ratio_threshold);
  vector<Match> initialMatches( matcher.computeMatches() );


  // Match keypoints.
  printStage("Filter matches by region growing robustly");
  timer.restart();
  vector<Region> regions;

  int num_region_growing = 2000;
  GrowthParams params;
  GrowMultipleRegions growRegions(initialMatches, params);
  regions = growRegions(num_region_growing, 0, &drawer);
  elapsed = timer.elapsedMs();
  cout << "Matching time = " << elapsed << " ms" << endl << endl;

  getKey();
}


// ========================================================================== //
// Main.
int main()
{
  imagePairMatching();

  return 0;
}