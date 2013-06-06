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

#ifndef DO_GRAPHICS_EVENTS_HPP
#define DO_GRAPHICS_EVENTS_HPP

#include <DO/Core.hpp>
#include <QEvent>

namespace DO {

  /*!
    \ingroup Graphics

    \defgroup Event Event handling functions
    \todo Investigate if it is useful. I don't remember.
    @{
   */

  enum EventType { 
    NO_EVENT,
    KEY_PRESSED = QEvent::KeyPress,
    KEY_RELEASED = QEvent::KeyRelease,
    MOUSE_PRESSED = QEvent::MouseButtonPress,
    MOUSE_RELEASED = QEvent::MouseButtonRelease,
    MOUSE_PRESSED_AND_MOVED = QEvent::MouseMove
  };

  struct Event
  {
    EventType type;
    int buttons;
    Point2i mousePos;
    int key;
    int keyModifiers;
  };

  inline Event noEvent()
  {
    Event e;
    e.type = NO_EVENT;
    e.key = Qt::Key_unknown;
    e.keyModifiers = Qt::NoModifier;
    e.buttons = Qt::NoButton;
    e.mousePos << -1, -1;
    return e;
  }

  inline Event keyPressed(int key, int keyModifiers = Qt::NoModifier)
  {
    Event e;
    e.type = KEY_PRESSED;
    e.key = key;
    e.keyModifiers = keyModifiers;
    e.buttons = Qt::NoButton;
    e.mousePos << -1, -1;
    return e;
  }

  inline Event keyReleased(int key, int keyModifiers = Qt::NoModifier)
  {
    Event e;
    e.type = KEY_RELEASED;
    e.key = key;
    e.keyModifiers = keyModifiers;
    e.buttons = Qt::NoButton;
    e.mousePos << -1, -1;
    return e;
  }

  inline Event mousePressed(int x, int y, int buttons,
                            int keyModifiers = Qt::NoModifier)
  {
    Event e;
    e.type = MOUSE_PRESSED;
    e.key = Qt::Key_unknown;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

  inline Event mouseReleased(int x, int y, int buttons,
                             int keyModifiers = Qt::NoModifier)
  {
    Event e;
    e.type = MOUSE_RELEASED;
    e.key = Qt::Key_unknown;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

  inline Event mouseMoved(int x, int y, int buttons,
                          int keyModifiers = Qt::NoModifier)
  {
    Event e;
    e.type = MOUSE_PRESSED_AND_MOVED;
    e.key = Qt::Key_unknown;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

}  /* namespace DO */

#endif /* DO_GRAPHICS_EVENTS_HPP */
