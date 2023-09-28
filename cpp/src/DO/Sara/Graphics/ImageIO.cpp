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
      gui_app(), "getImageFileFromDialogBox",
      Qt::BlockingQueuedConnection);

    return load(
      image,
      gui_app()->m_dialogBoxInfo.filename.toLocal8Bit().constData());
  }

  std::string select_video_file_from_dialog_box()
  {
    QMetaObject::invokeMethod(gui_app(), "getVideoFileFromDialogBox",
                              Qt::BlockingQueuedConnection);

    return gui_app()->m_dialogBoxInfo.filename.toLocal8Bit().constData();
  }

  bool save(const ImageView<unsigned char>& image, const std::string& name,
            int quality)
  {
    auto qimage = QImage(image.data(), image.width(), image.height(),
                         image.width(), QImage::Format_Indexed8);
    auto colorTable = QVector<QRgb>(256);
    for (int i = 0; i < 256; ++i)
      colorTable[i] = qRgb(i, i, i);
    qimage.setColorTable(colorTable);
    return qimage.save(QString(name.c_str()), 0, quality);
  }

  bool save(const ImageView<Rgb8>& image, const std::string& name, int quality)
  {
    return QImage(reinterpret_cast<const unsigned char*>(image.data()),
                  image.width(), image.height(), image.width() * 3,
                  QImage::Format_RGB888)
        .save(QString(name.c_str()), 0, quality);
  }

} /* namespace Sara */
} /* namespace DO */
