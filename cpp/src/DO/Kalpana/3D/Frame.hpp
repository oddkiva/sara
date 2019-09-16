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

#include <QtOpenGL>


class GLUquadric;


namespace DO { namespace Kalpana {

  const GLfloat red[]   = { 1.0f, 0.0f, 0.0f, 1.0f };
  const GLfloat green[] = { 0.0f, 1.0f, 0.0f, 1.0f };
  const GLfloat blue[]  = { 0.0f, 0.0f, 1.0f, 1.0f };
  const GLfloat yellow[]= { 1.0f, 1.0f, 0.0f, 1.0f };
  const GLfloat white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  const GLfloat defaultAmbient[] = { 0.2f, 0.2f, 0.2f, 1.0f };

  const double defaultAxisLength = 10.0;
  const double defaultAxisRadius = 0.125;

  class Frame
  {
  public:
    //! @brief Default constructor.
    Frame();

    //! @brief Destructor.
    ~Frame();

    //! @brief Draw frame in the window.
    void draw(double axisLength,
              double axisRadius = defaultAxisRadius);

  private:
    GLUquadric *pQuadObj;
  };

} /* namespace Kalpana */
} /* namespace DO */


#endif /* DO_KALPANA_GRAPHICS_FRAME_HPP */
