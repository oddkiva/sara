// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP
#define DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP

class QPolygonF;
class QImage;

namespace DO {

  /*!
    \ingroup Graphics
    \defgroup Draw2D Drawing 2D
    @{
  */

	// ======================================================================== //
	// Drawing commands
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in] x,y coordinates.
    @param[in] c   RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void drawPoint(int x, int y, const Color3ub& c);
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in] x,y coordinates.
    @param[in] c RGBA color in \f$[0, 255]^4\f$.
   */
  DO_EXPORT
	void drawPoint(int x, int y, const Color4ub& c);
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in] p coordinates where scalar is of float type.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
  void drawPoint(const Point2f& p, const Color3ub& c);
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] xc,yc circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
	void drawCircle(int xc, int yc, int r, const Color3ub& c, int penWidth = 1);
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] center circle center.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
	inline void drawCircle(const Point2i& center, int r, const Color3ub& c,
						             int penWidth = 1)
	{ drawCircle(center.x(), center.y(), r, c, penWidth); }
  /*!
    \brief Draw an axis-aligned ellipse in the active PaintingWindow window.
    @param[in] x,y,w,h bounding box parameters of the ellipse.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
	void drawEllipse(int x, int y, int w, int h, const Color3ub&c,
					         int penWidth = 1);
  /*!
    \brief Draw an oriented ellipse in the active PaintingWindow window.
    @param[in] center ellipse center.
    @param[in] r1,r2 ellipse radii.
    @param[in] degree ellipse orientation in degree.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
  void drawEllipse(const Point2f& center, float r1, float r2, float degree,
                   const Color3ub& c, int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] x1,y1,x2,y2 start and end points of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
	void drawLine(int x1, int y1, int x2, int y2, const Color3ub& c,
				        int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1,p2 start and end points of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
  void drawLine(const Point2f& p1, const Point2f& p2, const Color3ub& c,
                int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1,p2 start and end points of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
  void drawLine(const Point2d& p1, const Point2d& p2, const Color3ub& c,
                int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1,p2 start and end points of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
	inline void drawLine(const Point2i& p1, const Point2i& p2, const Color3ub&c,
						           int penWidth = 1)
	{ drawLine(p1.x(), p1.y(), p2.x(), p2.y(), c, penWidth); }
  /*!
    \brief Draw a rectangle in the active PaintingWindow window.
    @param[in] x,y,w,h rectangle parameters.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
   */
  DO_EXPORT
	void drawRect(int x, int y, int w, int h, const Color3ub& c,
				        int penWidth = 1);
  /*!
    \brief Draw a polygon in the active PaintingWindow window.
    @param[in] poly polygon.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
   */
  DO_EXPORT
	void drawPoly(const QPolygonF& poly, const Color3ub& c, int width = 1);
  /*!
    \brief Draw a polygon in the active PaintingWindow window.
    @param[in] x,y array of vertices of the polygon.
    @param[in] n number of vertices in the polygon.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
   */
  DO_EXPORT
	void drawPoly(const int x[], const int y[], int n, const Color3ub& c,
				        int width = 1);
  /*!
    \brief Draw a polygon in the active PaintingWindow window.
    @param[in] p array of vertices of the polygon.
    @param[in] n number of vertices in the polygon.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
   */
  DO_EXPORT
	void drawPoly(const Point2i* p, int n, const Color3ub& c, int width = 1);
  /*!
    \brief Draw text in the active PaintingWindow window.
    @param[in] x,y array of vertices of the polygon.
    @param[in] text text.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
   */
  DO_EXPORT
	void drawString(int x, int y, const std::string &s, const Color3ub& c,
					        int fontSize = 12, double alpha = 0, bool italic = false,
					        bool bold = false, bool underlined = false);
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] a,b,c,d start and end points of the arrow.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] arrowWidth,arrowHeight arrow parameters.
    @param[in] width width of the contour.
   */
  DO_EXPORT
	void drawArrow(int a, int b, int c, int d, const Color3ub& col,
				         int arrowWidth = 8, int arrowHeight = 5, 
                 int style = 0, int width = 1);
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] x1,y1,x2,y2 start and end points of the arrow.
    @param[in] col RGB color in \f$[0, 255]^3\f$.
    @param[in] ta,tl arrow parameters.
    @param[in] style arrow style.
    @param[in] width width of the contour.
   */
	inline void drawArrow(int x1, int y1, int x2, int y2, const Color3ub&  col,
						            double ta, double tl, int style, int width) 
	{ 
		drawArrow(x1, y1, x2, y2, col,
				  int(tl*cos(ta*3.14/180)), int(2*tl*sin(ta*3.14/180)),
				  style, width);
	}

	// ======================================================================== //
	// Filling commands
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] x,y,w,h bounding box of the ellipse.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillEllipse(int x, int y, int w, int h, const Color3ub& c);
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] p,w,h bounding box of the ellipse.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
	inline void fillEllipse(const Point2i& p, int w, int h, const Color3ub&c)
	{ fillEllipse(p.x(), p.y(), w, h, c); }
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] p ellipse center.
    @param[in] rx,ry ellipse radii.
    @param[in] degree ellipse orientation in degree.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
  void fillEllipse(const Point2f& p, float rx, float ry, float degree,
                   const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] x,y,w,h rectangle parameters.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillRect(int x, int y, int w, int h, const Color3ub& c);
	/*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p,w,h rectangle parameters.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
	inline void fillRect(const Point2i& p, int w, int h, const Color3ub&c)
	{ fillRect(p.x(), p.y(), w, h, c); }
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] x,y circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillCircle(int x, int y, int r, const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
	inline void fillCircle(const Point2i& p, int r, const Color3ub& c)
	{ fillCircle(p.x(), p.y(), r, c); }
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
  void fillCircle(const Point2f& p, float r, const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillPoly(const QPolygonF& polygon, const Color3ub& c);
  /*!
    \brief Draw a color-filled polygon in the active PaintingWindow window.
    @param[in] x,y array of vertices.
    @param[in] n number of vertices.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillPoly(const int x[], const int y[], int n, const Color3ub& c);
  /*!
    \brief Draw a color-filled polygon in the active PaintingWindow window.
    @param[in] p array of vertices \f$(\mathbf{p}_i)_{1\leq i \leq n}\f$ where 
                 \f$\mathbf{p}_i = (p_{2i}, p_{2i+1}) \in R^2 \f$.
    @param[in] n number of vertices.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillPoly(const int p[], int n, const Color3ub& c);
	/*!
    \brief Draw a color-filled polygon in the active PaintingWindow window.
    @param[in] p array of vertices.
    @param[in] n number of vertices.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
   */
  DO_EXPORT
	void fillPoly(const Point2i *p, int n, const Color3ub& c);

	// ======================================================================== //
	// Image display commands
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] image image.
    @param[in] xoff, yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
  DO_EXPORT
  void display(const QImage& image, int xoff = 0, int yoff = 0,
				       double fact = 1.);
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] x,y position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
   */
  DO_EXPORT
	void putColorImage(int x, int y, const Color3ub *data, int w, int h,
					           double fact = 1.);
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] p position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
   */
	inline void putColorImage(const Point2i& p, const Color3ub *data, int w, int h,
							              double fact = 1.)
	{ putColorImage(p.x(), p.y(), data, w, h, fact); }
  /*!
    \brief Draw a grayscale image in the active PaintingWindow window.
    @param[in] x,y position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
   */
  DO_EXPORT
	void putGreyImage(int x, int y, const uchar *data, int w, int h,
					          double fact = 1.);
  /*!
    \brief Draw a grayscale image in the active PaintingWindow window.
    @param[in] p position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
   */
	inline void putGreyImage(const Point2i& p, const uchar *data, int w, int h,
							             double fact = 1.)
	{ putGreyImage(p.x(), p.y(), data, w, h, fact); }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] xoff,yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
	inline void display(const Image<Color3ub>& image, int xoff = 0, int yoff = 0,
						          double fact = 1.)
	{ putColorImage(xoff, yoff, image.data(), image.width(), image.height(), fact); }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
	inline void display(const Image<Color3ub>& image,
						          const Point2i& off = Point2i::Zero(), double fact = 1.)
	{ display(image, off.x(), off.y(), fact); }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
	inline void display(const Image<Rgb8>& image, int xoff = 0, int yoff = 0,
						          double fact = 1.)
	{
		putColorImage(xoff, yoff, reinterpret_cast<const Color3ub *>(image.data()),
					        image.width(), image.height(), fact);
	}
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
  inline void display(const Image<Rgb8>& image,
                      const Point2i& off, double fact = 1.)
  { display(image, off.x(), off.y(), fact); }
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] image image.
    @param[in] xoff,yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
	template <typename T>
	inline void display(const Image<T>& image, int xoff = 0, int yoff = 0,
						          double fact = 1.)
	{ display(image.convert<Rgb8>(), xoff, yoff, fact); }
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] image image.
    @param[in] xoff,yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
   */
	template <typename T>
	void displayThreeChannelColorImageAsIs(const Image<T>& image, int xoff = 0, 
										                     int yoff = 0, double fact = 1.)
	{
		Image<Rgb8> tmp(image.sizes());
		for (int y = 0; y < image.height(); ++y)
			for (int x = 0; x < image.width(); ++x)
				for (int i = 0; i < 3; ++i)
					tmp(x,y)[i] = uchar(getRescaledChannel64f(image(x,y)[i])*255.);
		display(tmp, xoff, yoff, fact);
	}
  //! \brief View an image in a new PaintingWindow window.
	template <typename T>
	void view(const Image<T>& I, const std::string& windowTitle = "DO++",
            bool close = true)
	{
		// Original image.
		QWidget *win = openWindow(I.width(), I.height(), windowTitle);
		setActiveWindow(win);
		display(I);
		click();
		if (close)
			closeWindow(win);
	}

	// ======================================================================== //
	// Clearing commands
  //! \brief Clear the image
  DO_EXPORT
  void clearWindow();

  // ======================================================================== //
  // Painting options commands
  //! \brief Activate anti-aliased drawing.
  DO_EXPORT
  void setAntialiasing(Window w = activeWindow(), bool on = true);
  //! \bug Buggy. Investigate...
  DO_EXPORT
  void setTransparency(Window w = activeWindow(), bool on = true);

  // ======================================================================== //
  // Save screen command on window.
  //! \bried Save contents on the screen.
  DO_EXPORT
  bool saveScreen(Window w, const std::string& fileName);

  //! @}

} /* namespace DO */

#endif /* DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP */