// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QtGui>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  auto draw_point(ImageView<Rgb8>& image, int x, int y, const Rgb8& c) -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(to_QColor(c));
    p.drawPoint(x, y);
  }

  auto draw_line(ImageView<Rgb8>& image, int x1, int y1, int x2, int y2,
                 const Rgb8& c, int pen_width, bool antialiasing) -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setRenderHints(QPainter::Antialiasing, antialiasing);
    p.setPen(QPen(to_QColor(c), pen_width));
    p.drawLine(x1, y1, x2, y2);
  }

  auto draw_rect(ImageView<Rgb8>& image, int x, int y, int w, int h,
                 const Rgb8& c, int pen_width) -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setPen(QPen(to_QColor(c), pen_width));
    p.drawRect(x, y, w, h);
  }

  auto draw_circle(ImageView<Rgb8>& image, int xc, int yc, int r, const Rgb8& c,
                   int pen_width, bool antialiasing) -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.setRenderHints(QPainter::Antialiasing, antialiasing);
    p.setPen(QPen(to_QColor(c), pen_width));
    p.drawEllipse(QPoint(xc, yc), r, r);
  }

  auto draw_ellipse(ImageView<Rgb8>& image, const Eigen::Vector2f& center,
                    float r1, float r2, float degree, const Rgb8& c,
                    int pen_width, bool antialiasing) -> void
  {
    auto qimage = as_QImage(image);

    QPainter p(&qimage);
    p.setRenderHints(QPainter::Antialiasing, antialiasing);
    p.setPen(QPen(to_QColor(c), pen_width));
    p.translate(QPointF(center.x(), center.y()));
    p.rotate(degree);
    p.translate(-r1, -r2);
    p.drawEllipse(QRectF(0, 0, 2 * r1, 2 * r2));
  }

  auto draw_arrow(ImageView<Rgb8>& image, int x1, int y1, int x2, int y2,
                  const Rgb8& col, int pen_width, int arrow_width,
                  int arrow_height, int style, bool antialiasing) -> void
  {
    auto qimage = as_QImage(image);

    QPainter p(&qimage);
    p.setRenderHints(QPainter::Antialiasing, antialiasing);

    const auto qcol = to_QColor(col);
    auto sl = qreal{};
    const qreal dx = x2 - x1;
    const qreal dy = y2 - y1;
    const qreal norm = qSqrt(dx * dx + dy * dy);
    if (norm < 0.999)  // null vector
    {
      p.setPen(QPen(qcol, pen_width));
      p.drawPoint(x1, y1);
      return;
    }

    auto path = QPainterPath{};
    auto pts = QPolygonF{};

    const auto dx_norm = dx / norm;
    const auto dy_norm = dy / norm;
    const auto p1x =
        x1 + dx_norm * (norm - arrow_width) + arrow_height / 2. * dy_norm;
    const auto p1y =
        y1 + dy_norm * (norm - arrow_width) - arrow_height / 2. * dx_norm;
    const auto p2x =
        x1 + dx_norm * (norm - arrow_width) - arrow_height / 2. * dy_norm;
    const auto p2y =
        y1 + dy_norm * (norm - arrow_width) + arrow_height / 2. * dx_norm;
    switch (style)
    {
    case 0:
      p.setPen(QPen(qcol, pen_width));
      p.drawLine(x1, y1, x2, y2);
      p.drawLine(x2, y2, int(p1x), int(p1y));
      p.drawLine(x2, y2, int(p2x), int(p2y));
      break;
    case 1:
      pts << QPointF(p2x, p2y);
      pts << QPointF(x2, y2);
      pts << QPointF(p1x, p1y);
      sl = norm - (arrow_width * .7);
      pts << QPointF(x1 + dx_norm * sl + dy_norm * pen_width,
                     y1 + dy_norm * sl - dx_norm * pen_width);
      pts << QPointF(x1 + dy_norm * pen_width, y1 - dx_norm * pen_width);
      pts << QPointF(x1 - dy_norm * pen_width, y1 + dx_norm * pen_width);
      pts << QPointF(x1 + dx_norm * sl - dy_norm * pen_width,
                     y1 + dy_norm * sl + dx_norm * pen_width);
      path.addPolygon(pts);
      p.fillPath(path, qcol);
      break;
    case 2:
      pts << QPointF(p2x, p2y);
      pts << QPointF(x2, y2);
      pts << QPointF(p1x, p1y);
      sl = norm - arrow_width;
      pts << QPointF(x1 + dx_norm * sl + dy_norm * pen_width,
                     y1 + dy_norm * sl - dx_norm * pen_width);
      pts << QPointF(x1 + dy_norm * pen_width, y1 - dx_norm * pen_width);
      pts << QPointF(x1 - dy_norm * pen_width, y1 + dx_norm * pen_width);
      pts << QPointF(x1 + dx_norm * sl - dy_norm * pen_width,
                     y1 + dy_norm * sl + dx_norm * pen_width);
      path.addPolygon(pts);
      p.fillPath(path, qcol);
      break;
    default:
      break;
    }
  }

  auto draw_text(ImageView<Rgb8>& image, int x, int y, const std::string& text,
                 const Rgb8& color, int font_size, float orientation,
                 bool italic, bool bold, bool underline, int pen_width,
                 bool antialiasing) -> void
  {
    auto qimage = as_QImage(image);

    QPainter p(&qimage);
    p.setRenderHints(QPainter::Antialiasing, antialiasing);

    auto font = QFont{};
    font.setPointSize(font_size);
    font.setItalic(italic);
    font.setBold(bold);
    font.setUnderline(underline);

    auto textPath = QPainterPath{};
    const auto baseline = QPointF{0, 0};
    textPath.addText(baseline, font,
                     QString::QString::fromLocal8Bit(text.c_str()));

    // Outline the text by default for more visibility.
    p.setBrush(to_QColor(color));
    p.setPen(QPen(Qt::black, pen_width));
    p.setFont(font);

    p.translate(x, y);
    p.rotate(static_cast<qreal>(orientation));
    p.drawPath(textPath);
  }

  auto fill_rect(ImageView<Rgb8>& image, int x, int y, int w, int h,
                 const Rgb8& c) -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    p.fillRect(x, y, w, h, to_QColor(c));
  }

  auto fill_circle(ImageView<Rgb8>& image, int x, int y, int r, const Rgb8& c)
      -> void
  {
    QImage qimage(as_QImage(image));
    QPainter p(&qimage);
    QPainterPath path;
    path.addEllipse(QPointF(x, y), qreal(r), qreal(r));
    p.fillPath(path, to_QColor(c));
  }

}}  // namespace DO::Sara
