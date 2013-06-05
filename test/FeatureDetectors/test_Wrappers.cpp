#include <DO/FeatureDetectors.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace DO;
using namespace std;

void checkPatch(const Image<unsigned char>& image,
                const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("whole_picture.png"));


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);

  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.shapeMat() /= 4.;

  Matrix3f A;
  A.fill(0.f);
  A(0,0) = A(1,1) = r/2.;
  A(0,2) = k.feat().x();
  A(1,2) = k.feat().y();
  A(2,2) = 1.f;
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = float(x-r)/r;
      Point3f pp(u, v, 1.);
      pp = A*pp;

      Point2f p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = interpolate(image, p);
    }
  }

  Window w1 = activeWindow();
  Window w2 = openWindow(patchSz, patchSz);
  setActiveWindow(w2);
  setAntialiasing();
  display(patch);
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation(const Image<unsigned char>& image,
                           const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Yellow8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  Matrix2f Q = Rotation2D<float>(k.feat().orientation()).matrix();
  rg.shapeMat() = Q.transpose()*k.feat().shapeMat()*Q/4.;


  Matrix3f A(k.feat().affinity());  
  A.fill(0.f);
  A.block(0,0,2,2) = Q*r/2.;
  A(0,2) = k.feat().x();
  A(1,2) = k.feat().y();
  A(2,2) = 1.f;
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = float(x-r)/r;
      Point3f pp(u, v, 1.);
      pp = A*pp;

      Point2f p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = interpolate(image, p);
    }
  }

  Window w1 = activeWindow();
  Window w2 = openWindow(patchSz, patchSz);
  setActiveWindow(w2);
  setAntialiasing();
  display(patch);
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("rotated_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation2(const Image<unsigned char>& image,
                            const Keypoint& k)
{
  int w = image.width();
  int h = image.height();
  display(image);
  k.feat().drawOnScreen(Blue8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(k.feat());
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shapeMat() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3f A(k.feat().affinity());
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = 2*float(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = 2*float(x-r)/r;
      Point3f pp(u, v, 1.);
      pp = A*pp;

      Point2f p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = interpolate(image, p);
    }
  }

  Window w1 = activeWindow();
  Window w2 = openWindow(patchSz, patchSz);
  setActiveWindow(w2);
  setAntialiasing();
  display(patch);
  rg.drawOnScreen(Yellow8);
  saveScreen(activeWindow(), srcPath("normalized_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

const bool drawFeatureCenterOnly = false;
const Rgb8& c = Cyan8;

// ========================================================================== //
// Testing with painting
void testDoGSift(const Image<unsigned char>& image, bool drawFeatureCenterOnly = false)
{
  // Run DoG Detector
  cout << "Detecting DoG features... " << endl;
  HighResTimer t;
  double elapsed;
  t.restart();
  DoGSiftDetector dogsiftDetector;
  dogsiftDetector.setFirstOctave(-1);
  vector<Keypoint> keys(dogsiftDetector.run(image));
  elapsed = t.elapsedMs();
  cout << "Elapsed time = " << elapsed << " ms" << endl;
  cout << "Found " << keys.size() << " DoG-SIFT keypoints." << endl;

  cout << "Writing keypoints..." << endl;
  writeKeypoints(keys, srcPath("test.dogkey"));

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  cout << "done!" << endl;
  click();
}

void testHarAffSift(const Image<unsigned char>& image,
                    bool drawFeatureCenterOnly = false)
{
  // Run Harris Affine Detector
  cout << "Detecting Harris-Affine features... " << endl;
  vector<Keypoint> keys(HarAffSiftDetector().run(image, true, 100000));
  cout << "Found " << keys.size() << " Harris-Affine-SIFT keypoints" << endl;

  cout << "Writing keypoints..." << endl;
  writeKeypoints(keys, srcPath("test.haraffkey"));

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  //drawKeypoints(keys, Red8);
  for (int i = 0; i < keys.size(); ++i)
  {
    keys[i].feat().drawOnScreen(Red8);
    getKey();
  }
  cout << "done!" << endl;
  click();
}

void testHesAffSift(const Image<unsigned char>& image,
                    bool drawFeatureCenterOnly = false)
{
  // Run Hessian Affine Detector
  cout << "Detecting Hessian-Affine features... " << endl;
  vector<Keypoint> keys(HesAffSiftDetector().run(image, true, 200));
  cout << "Found " << keys.size() << " Hessian-Affine-SIFT keypoints" << endl;

  cout << "Writing keypoints..." << endl;
  writeKeypoints(keys, srcPath("test.hesaffkey"));

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  cout << "done!" << endl;
  click();
  
  checkPatch(image, keys[100]);
  checkAffineAdaptation(image, keys[100]);
  checkAffineAdaptation2(image, keys[100]);
}

void testMserSift(const Image<unsigned char>& image,
                  bool drawFeatureCenterOnly = false)
{
  // Run MSER Detector
  cout << "Detecting MSER features... " << endl;
  vector<Keypoint> keys(MserSiftDetector().run(image));
  cout << "Found " << keys.size() << " MSER-SIFT keypoints" << endl;

  cout << "Writing keypoints..." << endl;
  writeKeypoints(keys, srcPath("test.mserkey"));

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  drawKeypoints(keys, Red8);
  cout << "done!" << endl;
  click();
}

int main()
{
  Image<unsigned char> I;
  if (!load(I, srcPath("sunflowerField.jpg")))
    return -1;

  setActiveWindow(openWindow(I.width(), I.height()));
  setAntialiasing(activeWindow());
  testDoGSift(I);
  //testHarAffSift(I);
  //testHesAffSift(I);
  //testMserSift(I);
  getKey();

  return 0;
}