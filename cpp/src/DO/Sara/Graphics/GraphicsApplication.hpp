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

//! @file

#pragma once


namespace DO { namespace Sara {

  /*!
   *  @ingroup Graphics
   *  @defgroup GraphicsInternal Graphics Internals
   *  @brief This contains the Qt-based internal implementation of the Graphics
   *  module.
   *
   *  @{
   */

  //! @brief QApplication-derived class
  //! This graphic application establishes communication between the user
  //! drawing commands and the windows.
  class DO_SARA_EXPORT GraphicsApplication
  {
  public:
    class Impl;
    GraphicsApplication(int& argc, char** argv);
    ~GraphicsApplication();
    void register_user_main(int (*userMain)(int, char**));
    void register_user_main(std::function<int(int, char **)>);
    int exec();

  private:
    Impl* pimpl_;
  };

  //! @}

}}  // namespace DO::Sara


//! @addtogroup GraphicsInternal
//! @{

// ====================================================================== //
//! @brief Some convenient hacks.
int __main(int, char**);

#define GRAPHICS_MAIN()                                                        \
  int main(int argc, char** argv)                                              \
  {                                                                            \
    DO::Sara::GraphicsApplication app(argc, argv);                             \
    app.register_user_main(__main);                                            \
    return app.exec();                                                         \
  }                                                                            \
                                                                               \
  int __main(int, char**)


//! @}
