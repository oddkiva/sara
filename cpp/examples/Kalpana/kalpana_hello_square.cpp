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
using namespace std;


class SquareObject : public QObject
{
public:
  SquareObject(QObject* parent = nullptr)
    : QObject{parent}
  {
    initialize_shader_program();
    initialize_geometry();
    initialize_geometry_on_gpu();
  }

  ~SquareObject()
  {
    m_vao->release();
    m_vao->destroy();

    m_vbo.release();
    m_vbo.destroy();

    m_ebo.release();
    m_ebo.destroy();
  }

  void initialize_shader_program()
  {
    m_program = new QOpenGLShaderProgram{parent()};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
  }

  void initialize_geometry()
  {
    m_vertices = Tensor_<float, 2>{{4, 6}};
    m_vertices.flat_array() <<  //
      // coords            color
       0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  // top-right
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  // bottom-right
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  // bottom-left
      -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f;  // top-left

    m_triangles = Tensor_<unsigned int, 2>{{2, 3}};
    m_triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
  }

  void initialize_geometry_on_gpu()
  {
    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return offset * sizeof(float);
    };

    m_vao = new QOpenGLVertexArrayObject{parent()};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    if (!m_ebo.create())
      throw QException{};

    // Specify the vertex attributes here.
    m_vao->bind();

    // Copy the vertex data into the GPU buffer object.
    m_vbo.bind();
    m_vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vbo.allocate(m_vertices.data(),
                   static_cast<int>(m_vertices.size() * sizeof(float)));

    // Copy the triangles data into the GPU buffer object.
    m_ebo.bind();
    m_ebo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_ebo.allocate(m_triangles.data(),
                   static_cast<int>(m_triangles.size() * sizeof(unsigned int)));

    // Specify that the vertex shader param 0 corresponds to the first 3 float
    // data of the buffer object.
    m_program->enableAttributeArray(arg_pos["in_coords"]);
    m_program->setAttributeBuffer(
        /* location */ arg_pos["in_coords"],
        /* GL_ENUM */ GL_FLOAT,
        /* offset */ float_pointer(0),
        /* tupleSize */ 3,
        /* stride */ static_cast<int>(row_bytes(m_vertices)));

    // Specify that the vertex shader param 1 corresponds to the first 3 float
    // data of the buffer object.
    m_program->enableAttributeArray(arg_pos["in_color"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_color"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(3),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    // Unbind the vbo to protect its data.
    m_vao->release();
  }

  void render()
  {
    m_program->bind();
    m_vao->bind();
    glDrawElements(GL_TRIANGLES, m_triangles.size(), GL_UNSIGNED_INT, 0);
    m_program->release();
  }

protected:
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
    frag_color = vec4(out_color, 1.0);
  }
  )shader";

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"out_color", 0}};

  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};
};


class Window : public QOpenGLWindow
{
  SquareObject* m_square{nullptr};

public:
  Window() = default;

  void initializeGL() override
  {
    m_square = new SquareObject{context()};
  }

  void paintGL() override
  {
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_square->render();
  }
};


int main(int argc, char **argv)
{
  QGuiApplication app(argc, argv);
  QSurfaceFormat format;
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setVersion(3, 3);

  Window window;
  window.setFormat(format);
  window.resize(800, 600);
  window.show();

  return app.exec();
}
