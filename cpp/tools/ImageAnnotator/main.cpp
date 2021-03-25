// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2010-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <QApplication>

#include "MainWindow.hpp"


#ifdef WIN32
int WinMain(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
  QApplication app(argc, argv);

  MainWindow mainWin;
  mainWin.showMaximized();

  return app.exec();
}
