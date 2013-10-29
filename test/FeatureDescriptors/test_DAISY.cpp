#include <DO/FeatureDetectors.hpp>
#include <DO/FeatureDescriptors.hpp>
#include <DO/Graphics.hpp>

using namespace DO;
using namespace std;

Set<OERegion, RealDescriptor> computeDAISY(const Image<float>& image)
{
  // Time everything.
  HighResTimer timer;
  double elapsed = 0.;
  double DoGDetTime, oriAssignTime, daisyCompTime, gradGaussianTime;

  // We describe the work flow of the feature detection and description.
  Set<OERegion, RealDescriptor> keys;
  vector<OERegion>& DoGs = keys.features;
  DescriptorMatrix<float>& daisies = keys.descriptors;

  // 1. Feature extraction.
  printStage("Computing DoG extrema");
  timer.restart();
  ImagePyramidParams pyrParams(0);
  ComputeDoGExtrema computeDoGs(pyrParams);
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
  const ImagePyramid<float>& gaussPyr = computeDoGs.gaussians();
  gradG = gradPolar(gaussPyr);
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


  // 3. Feature description with DAISY
  printStage("Describe DoG extrema with DAISY descriptors");
  timer.restart();
  DAISY daisy;
  daisy.compute(daisies, DoGs, scaleOctPairs, gaussPyr);
  daisyCompTime = timer.elapsedMs();
  elapsed += daisyCompTime;
  cout << "description time = " << daisyCompTime << " ms" << endl;
  cout << "sifts.size() = " << daisies.size() << endl;
  // Summary in terms of computation time.
  printStage("Total Detection/Description time");
  cout << "DAISY computation time = " << elapsed << " ms" << endl;


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
      if (!isfinite(descriptors[i](j)))
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

  printStage("Detecting DoG-DAISY features");
  Set<OERegion, RealDescriptor> DoGDaisies = computeDAISY(image.convert<float>());
  const vector<OERegion>& features = DoGDaisies.features;

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