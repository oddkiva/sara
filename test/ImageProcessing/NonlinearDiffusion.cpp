//=============================================================================
//
// nldiffusion_functions.cpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file nldiffusion_functions.cpp
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "NonlinearDiffusion.hpp"
#include <DO/ImageProcessing.hpp>
#ifdef _OPENMP
# include <omp.h>
#endif

using namespace std;

namespace DO { namespace AKAZE {

void Gaussian_2D_Convolution(const Image<float>& src, Image<float>& dst,
                             size_t ksize_x, size_t ksize_y,
                             const float sigma)
{
  applyGaussianFilter(dst, src, sigma);
}

template <typename T>
void applyScharrRowDerivativeFilter(Image<T>& dst, const Image<T>& src)
{
  typedef typename ColorTraits<T>::ChannelType S;
  S meanKernel[] = { S( 3), S(10), S(3) };
  S diffKernel[] = { S(-1),  S(0), S(1) };

  if (dst.sizes() != src.sizes())
    dst.resize(src.sizes());
  Image<T> tmp(src.sizes());
  applyFastColumnBasedFilter(dst, dst, meanKernel, 3);
  applyFastRowBasedFilter(dst, src, diffKernel, 3);
}

template <typename T>
void applyScharrColumnDerivativeFilter(Image<T>& dst, const Image<T>& src)
{
  typedef typename ColorTraits<T>::ChannelType S;
  S meanKernel[] = { S( 3), S(10), S(3) };
  S diffKernel[] = { S(-1),  S(0), S(1) };

  if (dst.sizes() != src.sizes())
    dst.resize(src.sizes());
  Image<T> tmp(src.sizes());
  applyFastRowBasedFilter(dst, src, meanKernel, 3);
  applyFastColumnBasedFilter(dst, dst, diffKernel, 3);
}

void scharr_throw()
{
  cerr << "Cannot apply scharr derivatives in this manner" << endl;
  throw 0;
}

void Image_Derivatives_Scharr(const Image<float>& src, Image<float>& dst,
                              size_t xorder, size_t yorder)
{
  if (xorder != 0 && xorder != 1)
    scharr_throw();
  if (yorder != 0 && yorder != 1)
  scharr_throw();
  if (xorder == 1 && yorder == 0)
    applyScharrColumnDerivativeFilter(dst, src);
  else if (xorder == 0 && yorder == 1)
    applyScharrRowDerivativeFilter(dst, src);
  else
    scharr_throw();
}

void PM_G1(const Image<float>& Lx, const Image<float>& Ly, Image<float>& dst,
           const float k)
{
  dst.array() = - (Lx.array().square() + Ly.array().square()).exp() / (k*k);
}

void PM_G2(const Image<float>& Lx, const Image<float>& Ly, Image<float>& dst, const float k)
{
  dst.array() = ( (Lx.array().square()+Ly.array().square() + 1)/(k*k) ).inverse();
}

void Weickert_Diffusivity(const Image<float>& Lx, const Image<float>& Ly,
                          Image<float>& dst, const float k)
{
    // Gradient magnitude...
    dst.array() = Lx.array().square()+Ly.array().square();
    // ... divided by $k^2$, powered by $4$
    dst.array() = ( dst.array() / (k*k) ).pow(4);
    // ... inverse, multiplied by $-3.315$, exponentiated
    dst.array() = (dst.array().inverse() * -3.315f).exp();
    // ... and:
    dst.array() = 1.0 - dst.array();
}

//float Compute_K_Percentile(const cv::Mat &img, float perc,
//                           float gscale, size_t nbins,
//                           size_t ksize_x, size_t ksize_y)
//{
//  size_t nbin = 0, nelements = 0, nthreshold = 0, k = 0;
//  float kperc = 0.0, modg = 0.0, lx = 0.0, ly = 0.0;
//  float npoints = 0.0;
//  float hmax = 0.0;
//
//  // Create the array for the histogram and set the histogram to zero,
//  // just in case.
//  vector<float> hist(nbins, 0.f);
//
//  // Create the matrices
//  cv::Mat gaussian = cv::Mat::zeros(img.rows,img.cols,CV_32F);
//  cv::Mat Lx = cv::Mat::zeros(img.rows,img.cols,CV_32F);
//  cv::Mat Ly = cv::Mat::zeros(img.rows,img.cols,CV_32F);
//
//  // Perform the Gaussian convolution
//  Gaussian_2D_Convolution(img,gaussian,ksize_x,ksize_y,gscale);
//
//  // Compute the Gaussian derivatives Lx and Ly
//  Image_Derivatives_Scharr(gaussian,Lx,1,0);
//  Image_Derivatives_Scharr(gaussian,Ly,0,1);
//
//  // Skip the borders for computing the histogram
//  for( int i = 1; i < gaussian.rows-1; i++ )
//  {
//    for( int j = 1; j < gaussian.cols-1; j++ )
//    {
//      lx = *(Lx.ptr<float>(i)+j);
//      ly = *(Ly.ptr<float>(i)+j);
//      modg = sqrt(lx*lx + ly*ly);
//
//      // Get the maximum
//      if( modg > hmax )
//        hmax = modg;
//    }
//  }
//
//  // Skip the borders for computing the histogram
//  for( int i = 1; i < gaussian.rows-1; i++ )
//  {
//    for( int j = 1; j < gaussian.cols-1; j++ )
//    {
//      lx = *(Lx.ptr<float>(i)+j);
//      ly = *(Ly.ptr<float>(i)+j);
//      modg = sqrt(lx*lx + ly*ly);
//
//      // Find the correspondent bin
//      if( modg != 0.0 )
//      {
//        nbin = floor(nbins*(modg/hmax));
//
//        if( nbin == nbins )
//          nbin--;
//
//        hist[nbin]++;
//        npoints++;
//      }
//    }
//  }
//
//  // Now find the perc of the histogram percentile
//  nthreshold = (size_t)(npoints*perc);
//
//  for( k = 0; nelements < nthreshold && k < nbins; k++)
//    nelements = nelements + hist[k];
//
//  if( nelements < nthreshold )
//    kperc = 0.03;
//  else
//    kperc = hmax*((float)(k)/(float)nbins);	
//
//  return kperc;
//}
//
//void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst,
//                                const int xorder, const int yorder,
//                                const int scale )
//{
//   cv::Mat kx, ky;
//   Compute_Deriv_Kernels(kx, ky, xorder,yorder,scale);
//   cv::sepFilter2D(src,dst,CV_32F,kx,ky);
//}
//
//void NLD_Step_Scalar(cv::Mat &Ld, const cv::Mat &c, cv::Mat &Lstep, float stepsize)
//{
//#ifdef _OPENMP
//# pragma omp parallel for schedule(dynamic)
//#endif
//  for( int i = 1; i < Lstep.rows-1; i++ )
//  {
//    for( int j = 1; j < Lstep.cols-1; j++ )
//    {
//      float xpos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i)+j+1)))*((*(Ld.ptr<float>(i)+j+1))-(*(Ld.ptr<float>(i)+j)));
//      float xneg = ((*(c.ptr<float>(i)+j-1))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i)+j-1)));
//
//      float ypos = ((*(c.ptr<float>(i)+j))+(*(c.ptr<float>(i+1)+j)))*((*(Ld.ptr<float>(i+1)+j))-(*(Ld.ptr<float>(i)+j)));
//      float yneg = ((*(c.ptr<float>(i-1)+j))+(*(c.ptr<float>(i)+j)))*((*(Ld.ptr<float>(i)+j))-(*(Ld.ptr<float>(i-1)+j)));
//
//      *(Lstep.ptr<float>(i)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
//    }
//  }
//
//  for( int j = 1; j < Lstep.cols-1; j++ )
//  {
//    float xpos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j+1)))*((*(Ld.ptr<float>(0)+j+1))-(*(Ld.ptr<float>(0)+j)));
//    float xneg = ((*(c.ptr<float>(0)+j-1))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j-1)));
//
//    float ypos = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(1)+j)))*((*(Ld.ptr<float>(1)+j))-(*(Ld.ptr<float>(0)+j)));
//    float yneg = ((*(c.ptr<float>(0)+j))+(*(c.ptr<float>(0)+j)))*((*(Ld.ptr<float>(0)+j))-(*(Ld.ptr<float>(0)+j)));
//
//    *(Lstep.ptr<float>(0)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
//  }
//
//  for( int j = 1; j < Lstep.cols-1; j++ )
//  {
//    float xpos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j+1)))*((*(Ld.ptr<float>(Lstep.rows-1)+j+1))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
//    float xneg = ((*(c.ptr<float>(Lstep.rows-1)+j-1))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j-1)));
//
//    float ypos = ((*(c.ptr<float>(Lstep.rows-1)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-1)+j)));
//    float yneg = ((*(c.ptr<float>(Lstep.rows-2)+j))+(*(c.ptr<float>(Lstep.rows-1)+j)))*((*(Ld.ptr<float>(Lstep.rows-1)+j))-(*(Ld.ptr<float>(Lstep.rows-2)+j)));
//
//    *(Lstep.ptr<float>(Lstep.rows-1)+j) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
//  }
//
//  for( int i = 1; i < Lstep.rows-1; i++ )
//  {
//    float xpos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i)+1)))*((*(Ld.ptr<float>(i)+1))-(*(Ld.ptr<float>(i))));
//    float xneg = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i))));
//
//    float ypos = ((*(c.ptr<float>(i)))+(*(c.ptr<float>(i+1))))*((*(Ld.ptr<float>(i+1)))-(*(Ld.ptr<float>(i))));
//    float yneg = ((*(c.ptr<float>(i-1)))+(*(c.ptr<float>(i))))*((*(Ld.ptr<float>(i)))-(*(Ld.ptr<float>(i-1))));
//
//    *(Lstep.ptr<float>(i)) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
//  }
//
//  for( int i = 1; i < Lstep.rows-1; i++ )
//  {
//    float xpos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
//    float xneg = ((*(c.ptr<float>(i)+Lstep.cols-2))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-2)));
//
//    float ypos = ((*(c.ptr<float>(i)+Lstep.cols-1))+(*(c.ptr<float>(i+1)+Lstep.cols-1)))*((*(Ld.ptr<float>(i+1)+Lstep.cols-1))-(*(Ld.ptr<float>(i)+Lstep.cols-1)));
//    float yneg = ((*(c.ptr<float>(i-1)+Lstep.cols-1))+(*(c.ptr<float>(i)+Lstep.cols-1)))*((*(Ld.ptr<float>(i)+Lstep.cols-1))-(*(Ld.ptr<float>(i-1)+Lstep.cols-1)));
//
//    *(Lstep.ptr<float>(i)+Lstep.cols-1) = 0.5*stepsize*(xpos-xneg + ypos-yneg);
//  }
//
//  Ld = Ld + Lstep;
//}
//
//void Downsample_Image(const cv::Mat &src, cv::Mat &dst)
//{
//  int i1 = 0, j1 = 0, i2 = 0, j2 = 0;
//
//  for( i1 = 1; i1 < src.rows; i1+=2 )
//  {
//    j2 = 0;
//    for( j1 = 1; j1 < src.cols; j1+=2 )
//    {
//      *(dst.ptr<float>(i2)+j2) = 0.5*(*(src.ptr<float>(i1)+j1))+0.25*(*(src.ptr<float>(i1)+j1-1) + *(src.ptr<float>(i1)+j1+1));
//      j2++;
//    }
//
//    i2++;
//  }
//}
//
//void Halfsample_Image(const cv::Mat &src, cv::Mat &dst)
//{
//  // Make sure the destination image is of the right size
//  assert(src.cols/2==dst.cols);
//  assert(src.rows/2==dst.rows);
//  cv::resize(src,dst,dst.size(),0,0,cv::INTER_AREA);
//}
//
//void Compute_Deriv_Kernels(cv::OutputArray &kx_, cv::OutputArray &ky_,
//                           const int dx, const int dy, const int scale_)
//{
//  const int ksize = 3 + 2*(scale_-1);
//
//  if( scale_ == 1 )
//  {
//    // The usual Scharr kernel
//    cv::getDerivKernels(kx_,ky_,dx,dy,0,true,CV_32F);
//    return;
//  }
//
//  kx_.create(ksize,1,CV_32F,-1,true);
//  ky_.create(ksize,1,CV_32F,-1,true);
//  cv::Mat kx = kx_.getMat();
//  cv::Mat ky = ky_.getMat();
//
//  CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );
//
//  float w = 10.0/3.0;
//  float norm = 1.0/(2.0*scale_*(w+2.0));
//
//  for( int k = 0; k < 2; k++ )
//  {
//    cv::Mat* kernel = k == 0 ? &kx : &ky;
//    int order = k == 0 ? dx : dy;
//    //float kerI[ksize];
//    float kerI[10000];
//
//    for(int t = 0; t<ksize; t++)
//    {
//      kerI[t] = 0;
//    }
//
//    if( order == 0 )
//    {
//      kerI[0] = norm;
//      kerI[ksize/2] = w*norm;
//      kerI[ksize-1] = norm;
//    }
//    else if( order == 1 )
//    {
//      kerI[0] = -1;
//      kerI[ksize/2] = 0;
//      kerI[ksize-1] = 1;
//    }
//
//    cv::Mat temp(kernel->rows, kernel->cols, CV_32F, &kerI[0]);
//    temp.copyTo(*kernel);
//  }
//}

} /* namespace AKAZE */
} /* namespace DO */