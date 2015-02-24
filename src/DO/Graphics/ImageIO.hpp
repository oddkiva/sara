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
    convert(tmp, I);
    return true;
  }
  /*!
    \brief Load image from a dialog box.
    @param[in]  I image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
  bool load_from_dialog_box(Image<Rgb8>& I);
  /*!
    \brief Load image from a dialog box.
    @param[in]  I image file.
    \return true if image is successfully loaded.
    \return false otherwise.
   */
  DO_EXPORT
  bool load_from_dialog_box(Image<Color3ub>& I); 

  // ====================================================================== //
  // Image saving functions
  /*!
    \brief Save grayscale image.
    @param[in]  I image
    @param[in]  name path of the output image.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  inline bool save(const Image<unsigned char>& image, const std::string& name,
                   int quality = 85);
  /*!
    \brief Save color image.
    @param[in]  I image
    @param[in]  name path of the output image.
    @param[in]  quality image quality in \f$[0, 100]\f$.
    \return true if image is successfully saved.
    \return false otherwise.
   */
  inline bool save(const Image<Rgb8>& image, const std::string& name,
                   int quality = 85);

} /* namespace DO */

#endif /* DO_GRAPHICS_PAINTINGWINDOWCOMMANDS_HPP */