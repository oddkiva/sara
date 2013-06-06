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

#ifndef DO_GRAPHICS_IMAGEDRAW_HPP
#define DO_GRAPHICS_IMAGEDRAW_HPP

class QImage;

namespace DO {

  /*!
    \ingroup Graphics
    \defgroup ImageDrawing Drawing Image

    @{
   */

#ifdef USE_IMAGE_PAINTER
  // ====================================================================== //
  // A transformation stack management like OpenGL.
  class Painter : public QPainter
  {
    inline void beginImPaint(QImage& image) { begin(&image); }
    inline void endImPaint(QImage& image) { end(); }

    inline void pushImTransform(const QTransform& t)
    { transformStack.push(t); }
    inline void popImTransform()
    {
      QTransform newTransform( transform() * transformStack.pop().inverted() ); 
      setTransform(newTransform);
    }

    inline void imTranslate(double x, double y)
    { pushImTransform( QTransform::fromTranslate(qreal(x), qreal(y)) ); }
    inline void imScale(double sx, double sy)
    { pushImTransform( QTransform::fromScale(qreal(sx), qreal(sy)) ); }
    inline void imRotate(double degree)
    { QTransform r; r.rotate(degree); pushImTransform(r); }
    inline void imRotateRadians(double radian)
    { QTransform r; r.rotateRadians(radian); pushImTransform(r); }

  private:
    QStack<QTransform> transformStack;
  };
#endif

  /*!
    \brief Draw point on image.
    @param[in]  image image.
    @param[in]  x,y   coordinates.
    @param[in]  c     RGB color.
   */
  DO_EXPORT
  void drawPoint(Image<Rgb8>& image,
                 int x, int y, const Color3ub& c);
  /*!
    \brief Draw circle on image.
    @param[in]  image    image.
    @param[in]  xc,yc    circle center.
    @param[in]  r        circle radius.
    @param[in]  c        RGB color.
    @param[in]  penWidth width of the contour.
   */
  DO_EXPORT
  void drawCircle(Image<Rgb8>& image,
                  int xc, int yc, int r, const Color3ub& c, int penWidth = 1);
  /*!
    \brief Draw line on image.
    @param[in]  image       image.
    @param[in]  x1,y1,x2,y2 start and end points of the line.
    @param[in]  c           RGB color.
    @param[in]  penWidth    width of the contour.
   */
  DO_EXPORT
  void drawLine(Image<Rgb8>& image,
                int x1, int y1, int x2, int y2, const Color3ub& c,
                int penWidth = 1);
  /*!
    \brief Draw rectangle on image.
    @param[in]  image     image.
    @param[in]  x,y,w,h   start and end points of the line.
    @param[in]  c         RGB color.
    @param[in]  penWidth  width of the contour.
   */
  DO_EXPORT
  void drawRect(Image<Rgb8>& image,
                int x, int y, int w, int h, const Color3ub& c,
                int penWidth = 1);
  /*!
    \brief Draw color-filled rectangle on image.
    @param[in]  image     image.
    @param[in]  x,y,w,h   start and end points of the line.
    @param[in]  c         RGB color.
   */
  DO_EXPORT
  void fillRect(Image<Rgb8>& image,
                int x, int y, int w, int h, const Color3ub& c);
  /*!
    \brief Draw color-filled circle on image.
    @param[in]  image  image.
    @param[in]  x,y    circle center.
    @param[in]  r      circle radius.
    @param[in]  c      RGB color.
   */
  DO_EXPORT
  void fillCircle(Image<Rgb8>& image,
                  int x, int y, int r, const Color3ub& c);

  //! @}

} /* namespace DO */

#endif /* DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP */