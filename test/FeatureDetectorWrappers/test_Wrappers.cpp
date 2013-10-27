#include <DO/FeatureDetectorWrappers.hpp>
#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace DO;
using namespace std;

void checkPatch(const Image<unsigned char>& image,
                const OERegion& f)
{
  int w = image.width();
  int h = image.height();
  display(image);
  f.draw(Yellow8);
  saveScreen(activeWindow(), srcPath("whole_picture.png"));


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);

  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.shapeMat() /= 4.;

  Matrix3f A;
  A.fill(0.f);
  A(0,0) = A(1,1) = r/2.;
  A(0,2) = f.x();
  A(1,2) = f.y();
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
  rg.draw(Yellow8);
  saveScreen(activeWindow(), srcPath("patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation(const Image<unsigned char>& image,
                           const OERegion& f)
{
  int w = image.width();
  int h = image.height();
  display(image);
  f.draw(Yellow8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  Matrix2f Q = Rotation2D<float>(f.orientation()).matrix();
  rg.shapeMat() = Q.transpose()*f.shapeMat()*Q/4.;


  Matrix3f A(f.affinity());  
  A.fill(0.f);
  A.block(0,0,2,2) = Q*r/2.;
  A(0,2) = f.x();
  A(1,2) = f.y();
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
  rg.draw(Yellow8);
  saveScreen(activeWindow(), srcPath("rotated_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

void checkAffineAdaptation2(const Image<unsigned char>& image,
                            const OERegion& f)
{
  int w = image.width();
  int h = image.height();
  display(image);
  f.draw(Blue8);


  int r = 100;
  int patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);


  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shapeMat() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3f A(f.affinity());
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
  rg.draw(Yellow8);
  saveScreen(activeWindow(), srcPath("normalized_patch.png"));
  getKey();
  closeWindow(w2);

  milliSleep(40);
  setActiveWindow(w1);
}

template <typename Detector>
void testKeypointDetector(const Image<unsigned char>& image,
                          double detectorParam)
{
  printStage("Detecting features... ");
  Set<OERegion, RealDescriptor> keys;
  keys = Detector().run(image, true, detectorParam);
  cout << "Found " << keys.size() << " XXX-SIFT keypoints" << endl;

  display(image);
  printStage("Removing redundant features with the same descriptors.");
  cout << "// Only keep the ones with the high response threshold" << endl;
  removeRedundancies(keys.features, keys.descriptors);
  CHECK(keys.features.size());
  CHECK(keys.descriptors.size());

  // Draw features.
  printStage("Drawing features...");
  display(image);
  drawOERegions(keys.features, Green8);
  getKey();
}

int main()
{
  Image<unsigned char> I;
  if (!load(I, srcPath("obama_2.jpg")))
    return -1;

  setActiveWindow(openWindow(I.width(), I.height()));
  setAntialiasing(activeWindow());
  testKeypointDetector<HarAffSiftDetector>(I, 10000);
  testKeypointDetector<HesAffSiftDetector>(I, 200);
  testKeypointDetector<MserSiftDetector>(I, 0);
  getKey();

  return 0;
}