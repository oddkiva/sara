// ========================================================================== //
// This file is part of Kalpana.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_KALPANA_GRAPHICS_FRAME_HPP
#define DO_KALPANA_GRAPHICS_FRAME_HPP

#include <DO/Sara/Defines.hpp>

#include <QtOpenGL>


class GLUquadric;


namespace DO { namespace Kalpana {

  constexpr GLfloat red[] = {1.0f, 0.0f, 0.0f, 1.0f};
  constexpr GLfloat green[] = {0.0f, 1.0f, 0.0f, 1.0f};
  constexpr GLfloat blue[] = {0.0f, 0.0f, 1.0f, 1.0f};
  constexpr GLfloat yellow[] = {1.0f, 1.0f, 0.0f, 1.0f};
  constexpr GLfloat white[] = {1.0f, 1.0f, 1.0f, 1.0f};
  constexpr GLfloat default_ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};

  constexpr auto default_axis_length = 10.0;
  constexpr auto default_axis_radius = 0.125;

  class DO_SARA_EXPORT Frame
  {
  public:
    //! @brief Default constructor.
    Frame();

    //! @brief Destructor.
    ~Frame();

    //! @brief Draw frame in the window.
    void draw(double axisLength, double axisRadius = default_axis_radius);

  private:
    GLUquadric* pQuadObj;
  };

}}  // namespace DO::Kalpana


#endif /* DO_KALPANA_GRAPHICS_FRAME_HPP */
