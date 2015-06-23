// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP
#define DO_SARA_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP


namespace DO { namespace Sara {

  /*!
    \ingroup Graphics
    \defgroup Draw2D Drawing 2D
    @{
  */

  // ======================================================================== //
  // Drawing commands
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in]  x coordinates.
    @param[in]  y coordinates.
    @param[in]  c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_point(int x, int y, const Color3ub& c);
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in]  x coordinates.
    @param[in]  y coordinates.
    @param[in] c RGBA color in \f$[0, 255]^4\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_point(int x, int y, const Color4ub& c);
  /*!
    \brief Draw a point in the active PaintingWindow window.
    @param[in] p coordinates where each scalar is of float type.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_point(const Point2f& p, const Color3ub& c);
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] xc abscissa of the center.
    @param[in] yc ordinate of the center.
    @param[in] r radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.

   */
  DO_EXPORT
  bool draw_circle(int xc, int yc, int r, const Color3ub& c,
                   int penWidth = 1);
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] center circle center.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool draw_circle(const Point2i& center, int r, const Color3ub& c,
                          int penWidth = 1)
  { return draw_circle(center.x(), center.y(), r, c, penWidth); }
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] center circle center.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_circle(const Point2f& center, float r, const Color3ub& c,
                   int penWidth = 1);
  /*!
    \brief Draw a circle in the active PaintingWindow window.
    @param[in] center circle center.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_circle(const Point2d& center, double r, const Color3ub& c,
                   int penWidth = 1);
  /*!
    \brief Draw an axis-aligned ellipse in the active PaintingWindow window.
    @param[in] x x-coord of the top-left corner of the bounding box.
    @param[in] y y-coord of the top-left corner of the bounding box.
    @param[in] w width of the bounding box.
    @param[in] y height of the bounding box.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_ellipse(int x, int y, int w, int h, const Color3ub&c,
                    int penWidth = 1);
  /*!
    \brief Draw an oriented ellipse in the active PaintingWindow window.
    @param[in] center ellipse center.
    @param[in] r1 first ellipse radius.
    @param[in] r2 second ellipse radius.
    @param[in] degree ellipse orientation in degree.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_ellipse(const Point2f& center, float r1, float r2, float degree,
                    const Color3ub& c, int penWidth = 1);
  /*!
    \brief Draw an oriented ellipse in the active PaintingWindow window.
    @param[in] center ellipse center.
    @param[in] r1 first ellipse radius.
    @param[in] r2 second ellipse radius.
    @param[in] degree ellipse orientation in degree.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_ellipse(const Point2d& center, double r1, double r2, double degree,
                    const Color3ub& c, int penWidth);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] x1 start point of the line.
    @param[in] y1 start point of the line.
    @param[in] x2 end point of the line.
    @param[in] y2 end point of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_line(int x1, int y1, int x2, int y2, const Color3ub& c,
                 int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1 start point of the line.
    @param[in] p2 end point of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  inline bool draw_line(const Point2i& p1, const Point2i& p2, const Color3ub&c,
                        int penWidth = 1)
  { return draw_line(p1.x(), p1.y(), p2.x(), p2.y(), c, penWidth); }
    /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1 start point of the line.
    @param[in] p2 end point of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_line(const Point2f& p1, const Point2f& p2, const Color3ub& c,
                 int penWidth = 1);
  /*!
    \brief Draw a line in the active PaintingWindow window.
    @param[in] p1 start point of the line.
    @param[in] p2 end point of the line.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_line(const Point2d& p1, const Point2d& p2, const Color3ub& c,
                 int penWidth = 1);
  /*!
    \brief Draw a rectangle in the active PaintingWindow window.
    @param[in] x abscissa of the top-left corner of the rectangle.
    @param[in] y ordinate of the top-left corner of the rectangle.
    @param[in] w width of the rectangle.
    @param[in] h height of the rectangle.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_rect(int x, int y, int w, int h, const Color3ub& c,
                 int penWidth = 1);
  /*!
    \brief Draw a polygon in the active PaintingWindow window.
    @param[in] x,y array of vertices of the polygon.
    @param[in] n number of vertices in the polygon.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_poly(const int x[], const int y[], int n, const Color3ub& c,
                 int width = 1);
  /*!
    \brief Draw a polygon in the active PaintingWindow window.
    @param[in] p array of vertices of the polygon.
    @param[in] n number of vertices in the polygon.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_poly(const Point2i* p, int n, const Color3ub& c, int width = 1);
  /*!
    \brief Draw text in the active PaintingWindow window.
    @param[in] x,y array of vertices of the polygon.
    @param[in] text text.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] width width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_string(int x, int y, const std::string &s, const Color3ub& c,
                   int fontSize = 12, double alpha = 0, bool italic = false,
                   bool bold = false, bool underlined = false);
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] a,b,c,d start and end points of the arrow.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    @param[in] arrowWidth,arrowHeight arrow parameters.
    @param[in] width width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool draw_arrow(int a, int b, int c, int d, const Color3ub& col,
                  int arrowWidth = 8, int arrowHeight = 5,
                  int style = 0, int width = 1);
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] x1,y1,x2,y2 start and end points of the arrow.
    @param[in] col RGB color in \f$[0, 255]^3\f$.
    @param[in] ta,tl arrow parameters.
    @param[in] style arrow style.
    @param[in] width width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool draw_arrow(int x1, int y1, int x2, int y2, const Color3ub&  col,
                         double ta, double tl, int style, int width)
  {
    return draw_arrow(
      x1, y1, x2, y2, col,
      int(tl*cos(ta*3.14/180)), int(2*tl*sin(ta*3.14/180)),
      style, width);
  }
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] a start point of the arrow.
    @param[in] b end point of the arrow.
    @param[in] col RGB color in \f$[0, 255]^3\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool draw_arrow(const Point2f& a, const Point2f& b, const Color3ub& col,
                         int penWidth = 1)
  {
    return draw_arrow(
      int(a.x()), int(a.y()),
      int(b.x()), int(b.y()),
      col, 8, 5, 0, penWidth);
  }
  /*!
    \brief Draw an arrow in the active PaintingWindow window.
    @param[in] a start point of the arrow.
    @param[in] b end point of the arrow.
    @param[in] col RGBA color in \f$[0, 255]^4\f$.
    @param[in] penWidth width of the contour.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool draw_arrow(const Point2d& a, const Point2d& b, const Color3ub& col,
                         int penWidth = 1)
  {
    return draw_arrow(
      int(a.x()), int(a.y()),
      int(b.x()), int(b.y()),
      col, 8, 5, 0, penWidth);
  }

  // ======================================================================== //
  // Filling commands
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] x,y,w,h bounding box of the ellipse.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_ellipse(int x, int y, int w, int h, const Color3ub& c);
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] p,w,h bounding box of the ellipse.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool fill_ellipse(const Point2i& p, int w, int h, const Color3ub&c)
  { return fill_ellipse(p.x(), p.y(), w, h, c); }
  /*!
    \brief Draw a color-filled ellipse in the active PaintingWindow window.
    @param[in] p ellipse center.
    @param[in] rx,ry ellipse radii.
    @param[in] degree ellipse orientation in degree.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_ellipse(const Point2f& p, float rx, float ry, float degree,
                    const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] x,y,w,h rectangle parameters.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_rect(int x, int y, int w, int h, const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p,w,h rectangle parameters.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool fill_rect(const Point2i& p, int w, int h, const Color3ub&c)
  { return fill_rect(p.x(), p.y(), w, h, c); }
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] x,y circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_circle(int x, int y, int r, const Color3ub& c);
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool fill_circle(const Point2i& p, int r, const Color3ub& c)
  { return fill_circle(p.x(), p.y(), r, c); }
  /*!
    \brief Draw a color-filled circle in the active PaintingWindow window.
    @param[in] p circle center.
    @param[in] r circle radius.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_circle(const Point2f& p, float r, const Color3ub& c);
  /*!
    \brief Draw a color-filled polygon in the active PaintingWindow window.
    @param[in] x,y array of vertices.
    @param[in] n number of vertices.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_poly(const int x[], const int y[], int n, const Color3ub& c);
  /*!
    \brief Draw a color-filled polygon in the active PaintingWindow window.
    @param[in] p array of vertices.
    @param[in] n number of vertices.
    @param[in] c RGB color in \f$[0, 255]^3\f$.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool fill_poly(const Point2i *p, int n, const Color3ub& c);

  // ======================================================================== //
  // Image display commands
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] x,y position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool put_color_image(int x, int y, const Color3ub *data, int w, int h,
                       double fact = 1.);
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] p position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool put_color_image(const Point2i& p, const Color3ub *data, int w, int h,
                              double fact = 1.)
  { return put_color_image(p.x(), p.y(), data, w, h, fact); }
  /*!
    \brief Draw a grayscale image in the active PaintingWindow window.
    @param[in] x,y position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool put_grey_image(int x, int y, const unsigned char *data, int w, int h,
                      double fact = 1.);
  /*!
    \brief Draw a grayscale image in the active PaintingWindow window.
    @param[in] p position of the top-left image corner.
    @param[in] data color image.
    @param[in] w,h image sizes
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool put_grey_image(const Point2i& p, const unsigned char *data,
                             int w, int h, double fact = 1.)
  { return put_grey_image(p.x(), p.y(), data, w, h, fact); }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] xoff,yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool display(const Image<Color3ub>& image, int xoff = 0, int yoff = 0,
                      double fact = 1.)
  {
    return put_color_image(
      xoff, yoff, image.data(),
      image.width(), image.height(), fact);
  }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool display(const Image<Color3ub>& image,
                      const Point2i& off = Point2i::Zero(), double fact = 1.)
  { return display(image, off.x(), off.y(), fact); }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool display(const Image<Rgb8>& image, int xoff = 0, int yoff = 0,
                      double fact = 1.)
  {
    return put_color_image(
      xoff, yoff,
      reinterpret_cast<const Color3ub *>(image.data()),
      image.width(), image.height(), fact);
  }
  /*!
    \brief Draw a color image in the active PaintingWindow window.
    @param[in] image color image.
    @param[in] off position of the top-left image corner.
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  inline bool display(const Image<Rgb8>& image,
                      const Point2i& off, double fact = 1.)
  { return display(image, off.x(), off.y(), fact); }
  /*!
    \brief Draw an image in the active PaintingWindow window.
    @param[in] image image.
    @param[in] xoff,yoff position of the top-left image corner.
    @param[in] fact image viewing scale.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  template <typename T>
  inline bool display(const Image<T>& image, int xoff = 0, int yoff = 0,
                      double fact = 1.)
  { return display(image.template convert<Rgb8>(), xoff, yoff, fact); }
  /*!
    \brief View an image in a new PaintingWindow window.
    @param[in] image image.
    @param[in] windowTitle window title.
    @param[in] facto zoom factor.
   */
  template <typename T>
  void view_image(const Image<T>& image,
                  const std::string& windowTitle = "DO-CV",
                  double fact = 1.0)
  {
    QWidget *win = active_window() ? active_window() : 0;

    set_active_window(
      create_window(image.width()*fact, image.height()*fact, windowTitle));
    display(image, 0, 0, fact);
    get_key();
    close_window();

    if (win)
      set_active_window(win);
  }

  // ======================================================================== //
  // Clearing commands
  /*!
    \brief Clear the window contents.
    \return true if the command is issued on the window successfully.
    \return false otherwise.
   */
  DO_EXPORT
  bool clear_window();

  // ======================================================================== //
  // Painting options commands
  /*!
    \brief Activate anti-aliased drawing.
    @param[in] w a PaintingWindow instance.
    @param[in] on boolean value which activates or deactivates antialiasing.
    \return true if the antialiasing command is executed.
    \return false otherwise.
  */
  DO_EXPORT
  bool set_antialiasing(Window w = active_window(), bool on = true);
  /*!
    \bug Buggy. Investigate...
    @param[in] w a PaintingWindow instance.
    @param[in] on boolean value which activates or deactivates antialiasing.
    \return true if the transparency command is executed.
    \return false otherwise.
   */
  DO_EXPORT
  bool set_transparency(Window w = active_window(), bool on = true);

  // ======================================================================== //
  // Save screen command on window.
  /*!
    \brief Save contents on the screen.
    @param[in] w a PaintingWindow instance.
    @param[in] fileName a file name.
    \return true if save is successful.
    \return false otherwise.
   */
  DO_EXPORT
  bool save_screen(Window w, const std::string& fileName);

  //! @}

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP */
