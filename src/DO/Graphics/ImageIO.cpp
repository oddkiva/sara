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

#include <DO/Graphics.hpp>

namespace DO {

  // ====================================================================== //
  //! Image loading functions
  bool loadColorImage(const std::string& name, Color3ub *& data, 
                      int& w, int& h)
  {
    data = 0; w = 0; h = 0;
    QImage image(QString(name.c_str()));
    if (image.isNull())
      return false;
    image = image.convertToFormat(QImage::Format_RGB888);
    w = image.width(); h = image.height();
    data = new Color3ub[w*h];
    Color3ub *src = reinterpret_cast<Color3ub *>(image.bits());
    std::copy(src, src+w*h, data);
    return true;
  }

  bool loadGreyImage(const std::string& name, uchar *& data, 
                     int& w, int& h)
  {
    data = 0; w = 0; h = 0;
    QImage image(QString(name.c_str()));
    if (image.isNull())
      return false;
    w = image.width(); h = image.height();
    data = new uchar[w*h];
    if(!image.isGrayscale())
    {
      for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
          data[x+w*y] = uchar( qGray( image.pixel(x,y) ) );
    }
    else
    {
      for(int y = 0; y < h; ++y)
        for(int x = 0; x < w; ++x)
          data[x+w*y] = uchar( qRed( image.pixel(x,y) ) );
    }

    return true;
  }

  bool load(Image<Color3ub>& I, const std::string& name)
  {
    QImage image(QString(name.c_str()));
    if (image.isNull())
      return false;
    image = image.convertToFormat(QImage::Format_RGB888);
    I.resize(image.width(), image.height());
    
    Color3ub *dst = I.data();
    Color3ub *src = reinterpret_cast<Color3ub *>(image.bits());
    std::copy(src, src+image.width()*image.height(), dst);
    return true;
  }

  bool load(Image<Rgb8>& I, const std::string& name)
  {
    QImage image(QString(name.c_str()));
    if (image.isNull())
      return false;
    image = image.convertToFormat(QImage::Format_RGB888);
    I.resize(image.width(), image.height());
    Color3ub *dst = I.data();
    Color3ub *src = reinterpret_cast<Color3ub *>(image.bits());
    std::copy(src, src+image.width()*image.height(), dst);
    for (int y = 0; y < image.height(); ++y)
    {
      for (int x = 0; x < image.width(); ++x)
      {
        red(I(x,y)) = qRed(image.pixel(x,y));
        green(I(x,y)) = qGreen(image.pixel(x,y));
        blue(I(x,y)) = qBlue(image.pixel(x,y));
      }
    }
    return true;
  }

  bool loadFromDialogBox(Image<Rgb8>& I)
  {
    QMetaObject::invokeMethod(guiApp(), "getFileFromDialogBox",
                              Qt::BlockingQueuedConnection);
    bool r = load(I, guiApp()->interactiveBox.filename.toLocal8Bit().constData());
    return r;
  }

  bool loadFromDialogBox(Image<Color3ub>& I)
  {
    QMetaObject::invokeMethod(guiApp(), "getFileFromDialogBox",
                              Qt::BlockingQueuedConnection);
    bool r = load(I, std::string(guiApp()->interactiveBox.filename.toLocal8Bit().constData()));
    return r;
  }

  // ====================================================================== //
  //! Image saving functions
  bool saveColorImage(const std::string& name, const Color3ub *cols, 
                      int w, int h, int quality)
  {
    return QImage(reinterpret_cast<const uchar*>(cols),
                  w, h, w*3, QImage::Format_RGB888).
      save(QString(name.c_str()), 0, quality);
  }

  bool saveGreyImage(const std::string& name, const uchar *g, 
                     int w, int h, int quality)
  {
    QImage image(g, w, h, w, QImage::Format_Indexed8);
    QVector<QRgb> colorTable(256);
    for (int i = 0; i < 256; ++i)
      colorTable[i] = qRgb(i, i, i);
    image.setColorTable(colorTable);
    return image.save(QString(name.c_str()), 0, quality);
  }


} /* namespace DO */