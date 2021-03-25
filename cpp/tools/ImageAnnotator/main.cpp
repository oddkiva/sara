/*
 * =============================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Image annotator main entry point.
 *
 *        Version:  1.0
 *        Created:  16/06/2010 10:30:30
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#include <QApplication>
#include <QtGui>
#include "MainWindow.hpp"

//#ifdef WIN32
//int WinMain(int argc, char **argv)
//#else
int main(int argc, char **argv)
//#endif
{
  QApplication app(argc, argv);

  MainWindow *mainWin = new MainWindow;
  mainWin->showMaximized();

  return app.exec();
}