// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_GRAPHICS_USERTHREAD_HPP
#define DO_SARA_GRAPHICS_USERTHREAD_HPP

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

#include <DO/Sara/Defines.hpp>

#include "../Events.hpp"


namespace DO { namespace Sara {

  /*!
    \addtogroup GraphicsInternal

    @{
   */

  //! \brief This is actually where we actually call the drawing commands.
  class DO_EXPORT UserThread : public QThread
  {
    Q_OBJECT

  public:
    UserThread(QObject* parent = 0);

    void registerUserMain(int (*userMain)(int, char **)) { m_userMain = userMain; }

  public: /* timing methods */
    void milliSleep(int msec) { msleep(msec); }
    void microSleep(int usec) { usleep(usec); }

  public: /* I/O handling methods */
    int getMouse(int& x, int& y);
    int getKey();
    void getEvent(Event& e);

  public slots: /* I/O handling slot methods */
    void listenToWindowEvents();
    void pressedMouseButtons(int x, int y, Qt::MouseButtons buttons);
    void pressedKey(int key);
    void closedWindow();
    void receivedEvent(Event e);

  signals: /* for debugging and testing purposes */
    void sendEvent(QEvent *event, int delayMs);

  protected:
    void run();

  private:
    int (*m_userMain)(int, char **);

    QMutex m_mutex;
    QWaitCondition m_condition;

    bool m_doWaitForClick;
    Qt::MouseButtons m_mouseButton;
    int m_mouseX, m_mouseY;

    bool m_doWaitForKey;
    int m_key;

    Event m_event;
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS_USERTHREAD_HPP */
