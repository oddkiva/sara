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

#include <QDebug>

#include <DO/Sara/Graphics/DerivedQObjects/UserThread.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>


namespace DO { namespace Sara {

  UserThread::UserThread(QObject* parent)
    : QThread(parent)
    , m_userMain(0)
    , m_doWaitForClick(false)
    , m_doWaitForKey(false)
  {
  }

  int UserThread::getMouse(int& x, int& y)
  {
    m_mutex.lock();
    m_doWaitForClick = true;
    m_condition.wait(&m_mutex);
    m_mutex.unlock();
    x = m_mouseX;
    y = m_mouseY;
    if (m_mouseButton == Qt::LeftButton)
      return MOUSE_LEFT_BUTTON;
    else if (m_mouseButton == Qt::MiddleButton)
      return MOUSE_MIDDLE_BUTTON;
    else if (m_mouseButton == Qt::RightButton)
      return MOUSE_RIGHT_BUTTON;
    else
      return MOUSE_NO_BUTTON;
  }

  int UserThread::getKey()
  {
    m_mutex.lock();
    m_doWaitForKey = true;
    m_condition.wait(&m_mutex);
    m_mutex.unlock();
    return m_key;
  }

  void UserThread::listenToWindowEvents()
  {
    m_mutex.lock();
    m_condition.wait(&m_mutex);
    m_mutex.unlock();
  }

  void UserThread::getEvent(Event& e)
  {
    m_mutex.lock();
    e = m_event;
    m_mutex.unlock();
  }

  void UserThread::pressedMouseButtons(int x, int y, Qt::MouseButtons buttons)
  {
    m_mutex.lock();
    if (m_doWaitForClick)
    {
      m_doWaitForClick = false;
      m_mouseButton = buttons;
      m_mouseX = x;
      m_mouseY = y;
      m_condition.wakeOne();
    }
    m_mutex.unlock();
  }

  void UserThread::pressedKey(int key)
  {
    m_mutex.lock();
    if (m_doWaitForKey)
    {
      m_doWaitForKey = false;
      m_key = key;
      m_condition.wakeOne();
    }
    m_mutex.unlock();
  }

  void UserThread::receivedEvent(Event e)
  {
    m_mutex.lock();
    m_condition.wakeOne();
    m_event = e;
    m_mutex.unlock();
  }

  void UserThread::closedWindow()
  {
    // What to do in case the window was closed by the user and that
    // the program was expecting for a mouse click or a pressed key?
    m_mutex.lock();
    if (m_doWaitForClick || m_doWaitForKey)
    {
      qDebug() << "Window closed unexpectedly while waiting for user input!";
      m_condition.wakeOne();
    }
    // qDebug() << "Closing window as requested!";
    m_mutex.unlock();
  }

  void UserThread::run()
  {
    const auto& args = qApp->arguments();
    auto argc = static_cast<int>(args.size());
    auto argvVector = std::vector<char*>(args.size());
    std::transform(args.begin(), args.end(), argvVector.begin(),
                   [](const auto& arg) { return arg.toLocal8Bit().data(); });
    auto argv = &argvVector[0];

    if (m_userMain)
    {
      m_userMain(argc, argv);
      return;
    }
    else if (m_userMain2)
    {
      m_userMain2(argc, argv);
      return;
    }
    else
    {
      qFatal("No registered user main!");
      return;
    }
  }

}}  // namespace DO::Sara
