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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  bool load(Image<Color3ub>& I, const std::string& name)
  {
    QImage image{ QString(name.c_str()) };
    if (image.isNull())
      return false;
    image = image.convertToFormat(QImage::Format_RGB888);
    I.resize(image.width(), image.height());

    auto dst = I.data();
    auto src = reinterpret_cast<const Color3ub *>(image.bits());
    std::copy(src, src + image.width()*image.height(), dst);
    return true;
  }

  bool load(Image<Rgb8>& image, const std::string& name)
  {
    QImage qimage{ QString(name.c_str()) };
    if (qimage.isNull())
      return false;

    qimage = qimage.convertToFormat(QImage::Format_RGB32);
    if (qimage.format() != QImage::Format_RGB32)
      throw std::runtime_error("Failed to convert image to format RGB32");

    image.resize(qimage.width(), qimage.height());
    auto src = reinterpret_cast<const int *>(qimage.constBits());
    for (auto dst = image.data(); dst != image.end(); ++dst, ++src)
    {
      (*dst)[0] = *src >> 16;
      (*dst)[1] = *src >> 8;
      (*dst)[2] = *src;
    }
    return true;
  }

  bool load_from_dialog_box(Image<Rgb8>& image)
  {
    QMetaObject::invokeMethod(
      gui_app(), "getFileFromDialogBox",
      Qt::BlockingQueuedConnection);

    return load(
      image,
      gui_app()->m_dialogBoxInfo.filename.toLocal8Bit().constData());
  }

  bool load_from_dialog_box(Image<Color3ub>& image)
  {
    QMetaObject::invokeMethod(
      gui_app(), "getFileFromDialogBox",
      Qt::BlockingQueuedConnection);

    return load(
      image,
      std::string(gui_app()->m_dialogBoxInfo.filename.toLocal8Bit().constData()));
  }

  static
  bool saveColorImage(const std::string& name, const Color3ub *cols,
                      int w, int h, int quality)
  {
    return QImage(reinterpret_cast<const unsigned char*>(cols), w, h, w * 3,
                  QImage::Format_RGB888)
        .save(QString(name.c_str()), 0, quality);
  }

  static
  bool saveGreyImage(const std::string& name, const unsigned char *g,
                     int w, int h, int quality)
  {
    QImage image(g, w, h, w, QImage::Format_Indexed8);
    QVector<QRgb> colorTable(256);
    for (int i = 0; i < 256; ++i)
      colorTable[i] = qRgb(i, i, i);
    image.setColorTable(colorTable);
    return image.save(QString(name.c_str()), 0, quality);
  }

  bool save(const Image<unsigned char>& I, const std::string& name,
            int quality)
  {
    return saveGreyImage(name, I.data(), I.width(), I.height(), quality);
  }

  bool save(const Image<Rgb8>& I, const std::string& name, int quality)
  {
    return saveColorImage(name, I.data(), I.width(), I.height(), quality);
  }

} /* namespace Sara */
} /* namespace DO */
