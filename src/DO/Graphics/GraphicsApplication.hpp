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

#ifndef DO_GRAPHICS_GRAPHICSAPPLICATION_HPP
#define DO_GRAPHICS_GRAPHICSAPPLICATION_HPP

namespace DO {

  /*!
    \ingroup Graphics
    \defgroup GraphicsInternal Graphics Internals
    \brief This contains the Qt-based internal implementation of the Graphics 
    module.
  @{
 */
  
  //! \brief QApplication-derived class
  //! This graphic application establishes communication between the user 
  //! drawing commands and the windows.
  class DO_EXPORT GraphicsApplication
  {
  public:
    class Impl;
    GraphicsApplication(int argc, char **argv);
    ~GraphicsApplication();
    int exec();

  private:
    Impl *pimpl_;
  };
  
  //! @}

} /* namespace DO */

/*!
  \addtogroup GraphicsInternal

  @{
 */

// ====================================================================== //
//! \brief Some convenient hacks.
int __main(int, char **);

#define main(...)                           \
/*int*/ main(int argc, char **argv)         \
{                                           \
  DO::GraphicsApplication app(argc, argv);  \
  app.exec();                               \
}                                           \
                                            \
int __main(int argc, char **argv)


#endif /* DO_GRAPHICS_GRAPHICSAPPLICATION_HPP */