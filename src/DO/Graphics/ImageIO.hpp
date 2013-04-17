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

#ifndef DO_GRAPHICS_IMAGEIO_HPP
#define DO_GRAPHICS_IMAGEIO_HPP

class QImage;

namespace DO {

  /*!
    \ingroup Graphics

    \defgroup ImageIO Image I/O
    @{
   */

	// ====================================================================== //
	// Image loading functions
  /*!
    \brief Load color image.
    @param[in]  name path of the image file.
    @param[out] data array of \b Color3ub.
    @param[out] w,h  width and height of the image.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
	bool loadColorImage(const std::string& name, Color3ub *& data, 
                      int& w, int& h);
  /*!
    \brief Load grayscale image.
    @param[in]  name path of the image file.
    @param[out] data array of **unsigned char**.
    @param[out] w,h  width and height of the image.
    \return true if image is successfully loaded.
    \return false otherwise
   */
  DO_EXPORT
	bool loadGreyImage(const std::string& name, uchar *& data, 
                     int& w, int& h);
  /*!
    \brief Load color image.
    @param[out] I    color image with unsigned char channel type.
    @param[in]  name path of the image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
	bool load(Image<Color3ub>& I, const std::string& name);
  /*!
    \brief Load color image.
    @param[out] I    RGB image with unsigned char channel type.
    @param[in]  name path of the image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
	bool load(Image<Rgb8>& I, const std::string& name);
  /*!
    \brief Load image.
    @param[out] I    image where color is of type T.
    @param[in]  name path of the image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  template <typename T>
  bool load(Image<T>& I, const std::string& name)
  {
    Image<Rgb8> tmp;
    if (!load(tmp, name))
      return false;
    convert(I, tmp);
    return true;
  }
  /*!
    \brief Load image from a dialog box.
    @param[in]  I image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
	bool loadFromDialogBox(Image<Rgb8>& I);
	
  /*!
    \brief Load image from a dialog box.
    @param[in]  I image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
  bool loadFromDialogBox(Image<Color3ub>& I); 
  /*!
    \brief Load image from a dialog box.
    @param[in]  I image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  template <typename T>
  bool loadFromDialogBox(Image<T>& I)
  {
    Image<Rgb8> tmp;
    if (!loadFromDialogBox(tmp))
      return false;
    I = tmp.convert<T>();
    return true;
  }

	// ====================================================================== //
	// Image saving functions
  /*!
    \brief Save image.
    @param[in]  name path of the output image.
    @param[in]  cols array of \b Color3ub
    @param[in]  w image width.
    @param[in]  h image height.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  DO_EXPORT
	bool saveColorImage(const std::string& name, const Color3ub *cols, 
                      int w, int h, int quality = 85);
  /*!
    \brief Save grayscale image.
    @param[in]  name path of the output image.
    @param[in]  g array of **unsigned char**
    @param[in]  w image width.
    @param[in]  h image height.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  DO_EXPORT
	bool saveGreyImage(const std::string& name, const uchar *g, 
                     int w, int h, int quality = 85);
  /*!
    \brief Save grayscale image.
    @param[in]  I image
    @param[in]  name path of the output image.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  inline bool save(const Image<uchar>& I, const std::string& name,
                   int quality = 85)
  { return saveGreyImage(name, I.data(), I.width(), I.height(), quality); }
  /*!
    \brief Save color image.
    @param[in]  I image
    @param[in]  name path of the output image.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  inline bool save(const Image<Rgb8>& I, const std::string& name, int quality = 85)
  { return saveColorImage(name, I.data(), I.width(), I.height(), quality); }

} /* namespace DO */

#endif /* DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP */