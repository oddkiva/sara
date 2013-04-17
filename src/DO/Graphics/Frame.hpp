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

#ifndef DO_GRAPHICS_FRAME_HPP
#define DO_GRAPHICS_FRAME_HPP

#include <QtOpenGL>
#ifdef __APPLE__
# include <OpenGL/GLU.h>
#else
# include <GL/GLU.h>
#endif

namespace GLObject {

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
		Frame();
		~Frame();
    void draw(double axisLength, double axisRadius);
		static void genFrameList(GLuint *frameList,
								             double axisLength = defaultAxisLength,
								             double axisRadius = defaultAxisRadius);
	private:
		GLUquadricObj * pQuadObj;
	};


}

#endif /* DO_GRAPHICS_FRAME_HPP */
