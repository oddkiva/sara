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

#ifndef DO_GRAPHICS_GRAPHICSUTILITIES_HPP
#define DO_GRAPHICS_GRAPHICSUTILITIES_HPP

#include "DerivedQObjects/GraphicsApplication.hpp"

namespace DO {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  // ====================================================================== //
  // Convenience functions
  inline GraphicsApplication *guiApp()
  { return qobject_cast<DO::GraphicsApplication *>(qApp); }

  inline UserThread& userThread()
  { return guiApp()->userThread; }

  inline QWidget *activeWindow()
  { return guiApp()->activeWindow; }

  inline bool activeWindowIsVisible()
  { return guiApp()->activeWindowIsVisible(); }

  inline QStringList getMainArgs()
  { return guiApp()->arguments(); }

  //!@}

  // Deprecated since Qt 5.0.1.
  /*inline void getMainArgsFromGuiApp(int& argc, char **& argv)
  {
    argc = guiApp()->arguments().size();
    argv = guiApp()->argv();
  }*/

} /* namespace DO */


/*!
  \addtogroup GraphicsInternal

  @{
 */

// ====================================================================== //
//! \brief Some convenient hack macros.
int __main();

#define main()                              \
/*int*/ main(int argc, char **argv)         \
{                                           \
  DO::GraphicsApplication app(argc, argv);  \
  app.userThread.registerUserMain(__main);  \
  app.userThread.start();                   \
  app.exec();                               \
}                                           \
                                            \
int __main()

//! @}

#endif /* DO_GRAPHICS_GRAPHICSUTILITIES_HPP */