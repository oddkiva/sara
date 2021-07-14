// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/Tensor.hpp>

#include <QGuiApplication>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>
#include <QSurfaceFormat>
#include <QtCore/QException>


using namespace DO::Sara;


class TriangleWindow : public QOpenGLWindow
{
  QOpenGLShaderProgram* m_program{nullptr};

  const char* vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  out vec3 out_color;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
  }
  )shader";

  const char* fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;

    float dist = length(gl_PointCoord - vec2(0.5));

    if (dot(circCoord, circCoord) > 1.0)
        discard;
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    frag_color = vec4(out_color, alpha);
  }
  )shader";

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"out_color", 0}};

  Tensor_<float, 2> m_vertices;
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer* m_vbo{nullptr};

public:
  TriangleWindow() = default;

  ~TriangleWindow()
  {
    m_vao->release();
    m_vao->destroy();

    m_vbo->release();
    m_vbo->destroy();
    delete m_vbo;
  }

  void initializeGL() override
  {
    // Create the shader.
    m_program = new QOpenGLShaderProgram{context()};
    m_program->addCacheableShaderFromSourceCode(QOpenGLShader::Vertex,
                                                vertex_shader_source);
    m_program->addCacheableShaderFromSourceCode(QOpenGLShader::Fragment,
                                                fragment_shader_source);
    m_program->link();

    // Create the geometry data.
    m_vertices = Tensor_<float, 2>{{3, 6}};
    m_vertices.flat_array() <<  //
      // coords              color
      -0.5f, -0.5f, 0.0f,    1.0f, 0.0f, 0.0f,  // left
       0.5f, -0.5f, 0.0f,    0.0f, 1.0f, 0.0f,  // right
       0.0f,  0.5f, 0.0f,    0.0f, 0.0f, 1.0f;  // top

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return offset * sizeof(float);
    };


    // Transfer data from CPU to GPU.
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    m_vbo = new QOpenGLBuffer{};
    if (!m_vbo->create())
      throw QException{};

    m_program->bind();

    // Specify the vertex attributes here.
    {
      m_vao->bind();

      m_vbo->bind();
      m_vbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
      m_vbo->allocate(m_vertices.data(), m_vertices.size() * sizeof(float));

      // Specify that the vertex shader param 0 corresponds to the first 3 float
      // data of the buffer object.
      m_program->enableAttributeArray(arg_pos["in_coords"]);
      m_program->setAttributeBuffer(/* location */ arg_pos["in_coords"],
                                    /* GL_ENUM */ GL_FLOAT,
                                    /* offset */ float_pointer(0),
                                    /* tupleSize */ 3,
                                    /* stride */ row_bytes(m_vertices));

      // Specify that the vertex shader param 1 corresponds to the first 3 float
      // data of the buffer object.
      m_program->enableAttributeArray(arg_pos["in_color"]);
      m_program->setAttributeBuffer(/* location */ arg_pos["in_color"],
                                    /* GL_ENUM */ GL_FLOAT,
                                    /* offset */ float_pointer(3),
                                    /* tupleSize */ 3,
                                    /* stride */ row_bytes(m_vertices));
    }

    // Display options.
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
  }

  void paintGL() override
  {
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_program->bind();
    {
      m_vao->bind();
      glDrawArrays(GL_POINTS, 0, m_vertices.size(0));
    }
    m_program->release();
  }
};


int main(int argc, char **argv)
{
  QGuiApplication app(argc, argv);
  QSurfaceFormat format;
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setVersion(3, 3);

  TriangleWindow window;
  window.setFormat(format);
  window.resize(800, 600);
  window.show();

  //window.setAnimating(true);

  return app.exec();
}
