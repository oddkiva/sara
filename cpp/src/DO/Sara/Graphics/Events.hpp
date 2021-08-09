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

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  /*!
   *  @ingroup Graphics
   *
   *  @defgroup Event Event Handling Functions
   *  @todo Investigate if it is useful. I don't remember.
   *  @{
   */

  //! @brief I/O event types.
  enum class EventType : std::uint8_t
  {
    NO_EVENT,
    KEY_PRESSED,            /*! QEvent::KeyPress */
    KEY_RELEASED,           /*! QEvent::KeyRelease */
    MOUSE_PRESSED,          /*! QEvent::MouseButtonPress */
    MOUSE_RELEASED,         /*! QEvent::MouseButtonRelease */
    MOUSE_PRESSED_AND_MOVED /*! QEvent::MouseMove */
  };

  //! Key modifiers values (copy-pasted from Qt's documentation).
  enum KeyModifierType
  {
    KEY_NOMODIFIER  = 0x00000000,
    KEY_SHIFT       = 0x02000000,
    KEY_CTRL        = 0x04000000,
    KEY_ALT         = 0x08000000,
    KEY_META        = 0x10000000
  };

  //! Key type values (copy-pasted from Qt's documentation).
  enum KeyType
  {
    KEY_UNKNOWN   = 0x01ffffff,
    // Special keys.
    KEY_ESCAPE    = 0x01000000,
    KEY_TAB       =	0x01000001,
    KEY_BACKTAB   = 0x01000002,
    KEY_BACKSPACE = 0x01000003,
    KEY_RETURN    = 0x01000004,
    KEY_ENTER     = 0x01000005,
    KEY_INSERT    = 0x01000006,
    KEY_DELETE    = 0x01000007,
    KEY_PAUSE     = 0x01000008,
    KEY_PRINT     = 0x01000009,
    KEY_SYSREQ    = 0x0100000a,
    KEY_CLEAR     = 0x0100000b,
    KEY_HOME      = 0x01000010,
    KEY_END       = 0x01000011,
    // Arrow keys.
    KEY_LEFT      = 0x01000012,
    KEY_UP        = 0x01000013,
    KEY_RIGHT     = 0x01000014,
    KEY_DOWN      = 0x01000015,
    KEY_PAGEUP    = 0x01000016,
    KEY_PAGEDOWN  = 0x01000017
  };

  //! Mouse button values (copy-pasted from Qt's documentation).
  enum MouseButton
  {
    MOUSE_NO_BUTTON     = 0x00000000, /*! Qt::NoButton */
    MOUSE_LEFT_BUTTON   = 0x00000001, /*! Qt::LeftButton */
    MOUSE_RIGHT_BUTTON  = 0x00000002, /*! Qt::RightButton */
    MOUSE_MIDDLE_BUTTON = 0x00000004  /*! Qt::MiddleButton */
  };

  //! @brief I/O event structure.
  struct Event
  {
    EventType type = EventType::NO_EVENT;
    int buttons = MOUSE_NO_BUTTON;
    Point2i mousePos;
    int key = KEY_UNKNOWN;
    int keyModifiers = KEY_NOMODIFIER;
  };

  //! @brief Helper function to return a "no-event" Event.
  inline Event no_event()
  {
    Event e;
    e.key = KEY_UNKNOWN;
    e.keyModifiers = KEY_NOMODIFIER;
    e.buttons = MOUSE_NO_BUTTON;
    e.mousePos << -1, -1;
    return e;
  }

  //! @brief Helper function to return a "key-pressed" Event.
  inline Event key_pressed(int key, int keyModifiers = KEY_NOMODIFIER)
  {
    Event e;
    e.type = EventType::KEY_PRESSED;
    e.key = key;
    e.keyModifiers = keyModifiers;
    e.buttons = MOUSE_NO_BUTTON;
    e.mousePos << -1, -1;
    return e;
  }

  //! @brief Helper function to return a "key-released" Event.
  inline Event key_released(int key, int keyModifiers = KEY_NOMODIFIER)
  {
    Event e;
    e.type = EventType::KEY_RELEASED;
    e.key = key;
    e.keyModifiers = keyModifiers;
    e.buttons = MOUSE_NO_BUTTON;
    e.mousePos << -1, -1;
    return e;
  }

  //! @brief Helper function to return a "mouse-pressed" Event.
  inline Event mouse_pressed(int x, int y, int buttons,
                             int keyModifiers = KEY_NOMODIFIER)
  {
    Event e;
    e.type = EventType::MOUSE_PRESSED;
    e.key = KEY_UNKNOWN;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

  //! @brief Helper function to return a "mouse-pressed" Event.
  inline Event mouse_released(int x, int y, int buttons,
                              int keyModifiers = KEY_NOMODIFIER)
  {
    Event e;
    e.type = EventType::MOUSE_RELEASED;
    e.key = KEY_UNKNOWN;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

  //! @brief Helper function to return a "mouse-moved" Event.
  inline Event mouse_moved(int x, int y, int buttons,
                           int keyModifiers = KEY_NOMODIFIER)
  {
    Event e;
    e.type = EventType::MOUSE_PRESSED_AND_MOVED;
    e.key = KEY_UNKNOWN;
    e.keyModifiers = keyModifiers;
    e.buttons = buttons;
    e.mousePos << x, y;
    return e;
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */
