#include <DO/Graphics.hpp>
#include <DO/ImageProcessing.hpp>

using namespace DO;
using namespace std;

void createEllipse()
{
  openWindow(300, 300);
  fillRect(0, 0, 100, 100, Black8);
  saveScreen(activeWindow(), srcPath("ellipse.png"));
  getKey();
  closeWindow();
}

Image<float> warp(const Image<float>& I, const Matrix2f& T)
{
  Image<float> warpedI(300, 300);
  warpedI.array().fill(0.f);
  for (int y = 0; y < warpedI.height(); ++y)
  {
    for (int x = 0; x < warpedI.width(); ++x)
    {
      Point2f p(x,y);
      p = T*p;
      if ( p.x() < 0 && p.x() >= I.width()-1  ||
        p.y() < 0 && p.y() >= I.height()-1 )
        continue;
      warpedI(x,y) = interpolate(I, p);
    }
  }
  return warpedI;
}

int main()
{
  Image<float> I;
  if (!load(I, srcPath("ellipse.png")))
    return -1;
  //I = colorRescale(dericheBlur(I, 50.f));
  openWindow(I.width(), I.height());
  
  Matrix2f finalT;
  finalT.setIdentity();

  Image<float> oldI(I), diff(I);

  const int iter = 1000;
  for (int i = 0; i < iter; ++i)
  {
    // Check the image.
    display(I);
    getKey();
    
    diff.array() = I.array()-oldI.array();
    diff = colorRescale(diff);
    display(diff);
    getKey();

    // Compute the second order moment matrix.
    Image<Matrix2f> M(I.compute<Gradient>().compute<SecondMomentMatrix>());
    Matrix2f Sigma;
    for (Image<Matrix2f>::iterator M_it = M.begin(); M_it != M.end(); ++M_it)
      Sigma += *M_it;
    Sigma /= Sigma.norm();
      

    // Get the SVD decomposition of the second order moment matrix.
    JacobiSVD<Matrix2f> svd(Sigma, ComputeFullU);
    Vector2f sv = svd.singularValues();
    Matrix2f U = svd.matrixU();

    // Get one inverse transform.
    Vector2f radii( sv.cwiseSqrt().cwiseInverse() );
    Matrix2f T( U*radii.asDiagonal()*U.transpose() );
    T *= 1.f/radii(1);

    // Check.
    float angle =  atan2(U(1,0), U(0,0));
    angle = angle/(2*M_PI)*360.f;
    float rmin = 1.f/sqrt(sv(1));
    float rmax = 1.f/sqrt(sv(0));

    printStage("Iteration "+toString(i));
    cout << "Sigma = " << endl << Sigma << endl << endl;
    cout << "U*S*U^T = " << endl <<  U*sv.asDiagonal()*U.transpose() << endl << endl;
    cout << "T = " << endl << T << endl << endl;
    cout << "radii = " << radii.transpose() << endl << endl;
    cout << "scaleFactor= " << sqrt(sv(0)/sv(1)) << endl << endl;

    cout << "singular values = " << sv.transpose() << endl;
    cout << "U = " << endl << U << endl;
    cout << "angle = " << angle << " degrees" << endl;
    cout << "rmin = " << rmin << " rmax = " << rmax << endl;
    cout << "ratio = " << rmax/rmin << endl;

    oldI = I;
    I = warp(I, T);
  }

  return 0;
}
