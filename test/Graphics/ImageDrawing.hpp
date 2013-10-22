#include <DO/Core.hpp>


using namespace DO;


template <typename Color>
static inline void setColorSafely(Image<Color>& image, int x, int y,
                                  const Color& color)
{
  if (x < 0 || x >= image.width() || y < 0 || y >= image.height())
    return;
  image(x,y) = color;
}

struct Bresenham
{
  template <typename Color>
  static void drawLine(Image<Color>& image, int x0, int y0, int x1, int y1,
                       const Color& color)
  {
    const int dx = abs(x1-x0);
    const int dy = abs(y1-y0); 
    const int sx = x0 < x1 ? 1 : -1;
    const int sy = y0 < y1 ? 1 : -1;
    int err = dx-dy;

    while (true)
    {
      setColorSafely(image, x0, y0, color);

      if (x0 == x1 && y0 == y1)
        return;
      const int e2 = 2*err;

      if (e2 > -dy)
      {
        err -= dy;
        x0 += sx;
      }

      if (x0 == x1 && y0 == y1)
      {
        setColorSafely(image, x0, y0, color);
        return;
      }
      if (e2 < dx)
      {
        err += dx;
        y0 += sy;
      }
    }
  }

  template <typename Color>
  static void drawCircle(Image<Color>& image, int x1, int y1, int r,
                         const Color& color)
  {

  }

  template <typename Color>
  static void drawEllipse(Image<Color>& image, int x1, int y1, int r1, int r2,
                          const Color& color)
  {

  }
};

struct Wu
{
  static inline int trunc(float x) { return static_cast<int>(x); }
  static inline float frac(float x) { float res = 0.f; std::modf(res, &x); return res; }
  static inline float invfrac(float x) { return 1.f - frac(x); }

//  template <typename Color>
//  static void drawLine(Image<Color>& image, int x1, int y1, int x2, int y2,
//                       const Color& color)
//  {
//    using namespace std;
//
//    float grad, xd, yd, length,xm,ym, xgap, ygap, xend, yend, xf, yf, brightness1, brightness2;
//    int x, y, ix1, ix2, iy1, iy2;
//    unsigned char c1,c2;
//
//    // Delta
//    xd = (x2-x1);
//    yd = (y2-y1);
//
//    //check line gradient horizontal(ish) lines
//    if (abs(xd) > abs(yd)) 
//    {
//      if (x1 > x2)        // if line is back to front
//      {
//        swap(x1, x2);     // then swap it round
//        swap(y1, y2);
//        xd = (x2-x1);     // and recalc xd and yd
//        yd = (y2-y1);
//      }
//
//      grad = yd/xd;       // gradient of the line
//
//      // End Point 1
//      xend = trunc(x1+.5);          // find nearest integer X-coordinate
//      yend = y1 + grad*(xend-x1);   // and corresponding Y value
//
//      xgap = invfrac(x1+.5);        // distance i
//
//      ix1  = int(xend);             // calc screen coordinates
//      iy1  = int(yend);
//
//      brightness1 = invfrac(yend) * xgap; // calc the intensity of the other 
//      brightness2 =    frac(yend) * xgap; // end point pixel pair.
//
//      c1 = static_cast<unsigned char>(brightness1 * MaxPixelValue); // calc pixel values
//      c2 = static_cast<unsigned char>(brightness2 * MaxPixelValue);
//
//      setColorSafely(image, ix1, iy1  , c1);  // draw the pair of pixels
//      setColorSafely(image, ix1, iy1+1, c2);
//
//      yf = yend+grad;                         // calc first Y-intersection for
//                                              // main loop
//
//      //End Point 2
//
//      xend = trunc(x2+.5);                  // find nearest integer X-coordinate
//      yend = y2 + grad*(xend-x2);           // and corresponding Y value
//
//      xgap = invfrac(x2-.5);                // distance i
//
//      ix2  = int(xend);                     // calc screen coordinates
//      iy2  = int(yend);
//
//      brightness1 = invfrac(yend) * xgap;    // calc the intensity of the first 
//      brightness2 =    frac(yend) * xgap;    // end point pixel pair.
//
//      c1 = static_cast<unsigned char>(brightness1 * MaxPixelValue); // calc pixel values
//      c2 = static_cast<unsigned char>(brightness2 * MaxPixelValue);	
//
//      setColorSafely(image, ix2,iy2, c1);			// draw the pair of pixels
//      setColorSafely(image, ix2,iy2+1, c2);
//
//      // MAIN LOOP
//      for (x = ix1+1; x < ix2; ++x)
//      {
//        brightness1 = invfrac(yf);		        // calc pixel brightnesses
//        brightness2 =    frac(yf);
//
//        c1 = byte(brightness1 * MaxPixelValue);	// calc pixel values
//        c2 = byte(brightness2 * MaxPixelValue);	
//
//        setColorSafely(x,int(yf), c1);			// draw the pair of pixels
//        setColorSafely(x,int(yf)+1, c2);
//
//        yf = yf + grad;				// update the y-coordinate
//      }
//
//    }
//    else {
//      /*vertical(ish) lines
//
//      handle the vertical(ish) lines in the
//      same way as the horizontal(ish) ones
//      but swap the roles of X and Y*/
//    }
//  }

  template <typename Color>
  static void drawCircle(Image<Color>& image, int x1, int y1, int r,
                         const Color& color)
  {

  }

  template <typename Color>
  static void drawEllipse(Image<Color>& image, int x1, int y1, int r1, int r2,
                          const Color& color)
  {

  }
};