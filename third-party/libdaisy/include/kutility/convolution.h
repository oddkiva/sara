#ifndef KUTILITY_CONVOLUTION_H
#define KUTILITY_CONVOLUTION_H

#if defined(WITH_OPENCV) && defined(WITH_OPENCV_EXTRAS)
   #include "kutility/convolution_opencv.h"
#else
   #include "kutility/convolution_default.h"
#endif

namespace kutility
{
   template<class T>
   inline void convolve_sym( T* image, int h, int w, T* kernel, int ksize, T* out=NULL )
   {
      if( out == NULL ) out = image;
      else memcpy( out, image, sizeof(T)*h*w );

      if( h == 240 && w ==  320 ) { convolve_sym_(out, 240, 320, kernel, ksize); return; }
      if( h == 480 && w ==  640 ) { convolve_sym_(out, 480, 640, kernel, ksize); return; }
      if( h == 512 && w ==  512 ) { convolve_sym_(out, 512, 512, kernel, ksize); return; }
      if( h == 512 && w ==  768 ) { convolve_sym_(out, 512, 768, kernel, ksize); return; }
      if( h == 600 && w ==  800 ) { convolve_sym_(out, 600, 800, kernel, ksize); return; }
      if( h == 768 && w == 1024 ) { convolve_sym_(out, 768, 1024, kernel, ksize); return; }
      if( h == 1024 && w == 768 ) { convolve_sym_(out, 1024, 768, kernel, ksize); return; }
      if( h == 256 && w ==  256 ) { convolve_sym_(out, 256, 256, kernel, ksize); return; }
      if( h == 128 && w ==  128 ) { convolve_sym_(out, 128, 128, kernel, ksize); return; }
      if( h == 128 && w ==  192 ) { convolve_sym_(out, 128, 192, kernel, ksize); return; }
      cout<<"[convolve_sym] insert this h,w to unrolling list: "<<h<<" "<<w<<endl;
      convolve_sym_(out, h, w, kernel, ksize);
   }
}
#endif
