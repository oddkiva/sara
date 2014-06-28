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

#include "../GraphicsUtilities.hpp"
#include "UserThread.hpp"
#include <QDebug>

namespace DO {

  UserThread::UserThread(QObject* parent)
    : QThread(parent)
    , userMain_(0)
    , doWaitForClick_(false)
    , doWaitForKey_(false)
  {
  }

  int UserThread::getMouse(int& x, int& y)
  {
    mutex_.lock();
    doWaitForClick_ = true;
    condition_.wait(&mutex_);
    mutex_.unlock();
    x = mouseX_; y = mouseY_;
    if (mouseButton_ == Qt::LeftButton)
      return MOUSE_LEFT_BUTTON;
    else if (mouseButton_ == Qt::MiddleButton)
      return MOUSE_MIDDLE_BUTTON;
    else if (mouseButton_ == Qt::RightButton)
      return MOUSE_RIGHT_BUTTON;
    else
      return MOUSE_NO_BUTTON;
  }

  int UserThread::getKey()
  {
    mutex_.lock();
    doWaitForKey_ = true;
    condition_.wait(&mutex_);
    mutex_.unlock();
    return key_;
  }

  void UserThread::listenToWindowEvents()
  {
    mutex_.lock();
    condition_.wait(&mutex_);
    mutex_.unlock();
  }

  void UserThread::getEvent(Event& e)
  {
    mutex_.lock();
    e = event_;
    mutex_.unlock();
  }

  void UserThread::pressedMouseButtons(int x, int y, Qt::MouseButtons buttons)
  {
    mutex_.lock();
    if (doWaitForClick_)
    {
      doWaitForClick_ = false;
      mouseButton_ = buttons; mouseX_ = x; mouseY_ = y;
      condition_.wakeOne();
    }
    mutex_.unlock();
  }

  void UserThread::pressedKey(int key)
  {
    mutex_.lock();
    if (doWaitForKey_)
    {
      doWaitForKey_ = false;
      key_ = key;
      condition_.wakeOne();
    }
    mutex_.unlock();
  }

  void UserThread::receivedEvent(Event e)
  {
    mutex_.lock();
    condition_.wakeOne();
    event_ = e;
    mutex_.unlock();
  }

  void UserThread::closedWindow()
  {
    // What to do in case the window was closed by the user and that 
    // the program was expecting for a mouse click or a pressed key?
    mutex_.lock();
    if (doWaitForClick_ || doWaitForKey_)
    {
      qDebug() << "Window closed unexpectedly while waiting for user input!";
      condition_.wakeOne();
    }
    //qDebug() << "Closing window as requested!";
    mutex_.unlock();
  }

  void UserThread::run()
  {
    //qDebug("Running user thread!");
    if (!userMain_)
    {
      qFatal("No registered user main!");
      return;
    }
    int ret;
    ret = userMain_(getGuiApp()->argc, getGuiApp()->argv);
    return;
  }

} /* namespace DO */