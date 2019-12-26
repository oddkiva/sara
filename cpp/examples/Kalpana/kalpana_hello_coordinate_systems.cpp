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

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>

#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QtCore/QException>
#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLTexture>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QOpenGLWindow>

#include <map>


using namespace DO::Sara;
using namespace std;


const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec2 in_tex_coords;

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    gl_PointSize = 5.f;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";

const auto fragment_shader_source = R"shader(
#version 330 core
  in vec2 out_tex_coords;
  out vec4 frag_color;

  uniform sampler2D texture0;
  uniform sampler2D texture1;

  void main()
  {
    if (out_tex_coords.x > 0.5)
      frag_color = texture(texture0, out_tex_coords);
    else
      frag_color = texture(texture1, out_tex_coords);
  }
  )shader";

std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                      {"in_tex_coords", 1},  //
                                      {"out_color", 0}};


auto make_cube()
{
  auto cube = Tensor_<float, 2>{6 * 6, 5};
  cube.flat_array() <<
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
     0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
     0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
     0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
     0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f, 1.0f;
  return cube;
}


class Window : public QOpenGLWindow
{
private:
  QOpenGLShaderProgram* m_program{nullptr};
  QOpenGLDebugLogger* m_logger{nullptr};

  std::array<Vector3f, 10> m_cubePositions;
  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};

  QOpenGLTexture* m_texture0;
  QOpenGLTexture* m_texture1;

  QMatrix4x4 m_projection;
  QMatrix4x4 m_view;
  Timer timer;

public:
  Window() = default;

  ~Window()
  {
    makeCurrent();
    {
      m_vao->release();
      m_vao->destroy();

      m_vbo.release();
      m_vbo.destroy();

      m_texture0->release();
      m_texture0->destroy();
      delete m_texture0;

      m_texture1->release();
      m_texture1->destroy();
      delete m_texture1;
    }
    doneCurrent();
  }

  void initialize_shader_program()
  {
    SARA_DEBUG << "Initialize shader program" << std::endl;

    m_program = new QOpenGLShaderProgram{context()};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
    m_program->bind();
  }

  void initialize_geometry()
  {
    SARA_DEBUG << "Initialize geometry data" << std::endl;

    m_vertices = make_cube();

    m_cubePositions = {
      Vector3f( 0.0f,  0.0f,  0.0f), Vector3f( 2.0f,  5.0f, -15.0f),
      Vector3f(-1.5f, -2.2f, -2.5f), Vector3f(-3.8f, -2.0f, -12.3f),
      Vector3f( 2.4f, -0.4f, -3.5f), Vector3f(-1.7f,  3.0f, - 7.5f),
      Vector3f( 1.3f, -2.0f, -2.5f), Vector3f( 1.5f,  2.0f, - 2.5f),
      Vector3f( 1.5f,  0.2f, -1.5f), Vector3f(-1.3f,  1.0f, - 1.5f)
    };
  }

  void initialize_geometry_on_gpu()
  {
    SARA_DEBUG << "Initialize geometry data on GPU" << std::endl;
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };

    const auto float_pointer = [](int offset) {
      return offset * sizeof(float);
    };

    m_vao->bind();

    // Copy the vertex data into the GPU buffer object.
    m_vbo.bind();
    m_vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vbo.allocate(m_vertices.data(), m_vertices.size() * sizeof(float));

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    m_program->enableAttributeArray(arg_pos["in_coords"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_coords"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(0),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    // Texture coordinates.
    m_program->enableAttributeArray(arg_pos["in_tex_coords"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_tex_coords"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(3),
                                  /* tupleSize */ 2,
                                  /* stride */ row_bytes(m_vertices));

    m_vao->release();
  }

  void initialize_texture_on_gpu()
  {
    SARA_DEBUG << "Initialize texture data on GPU" << std::endl;

    // Texture 0.
    const auto image0_path = src_path("../../../data/ksmall.jpg");
    const auto image0 = QImage{image0_path}.mirrored();
    m_texture0 = new QOpenGLTexture{image0};
    m_texture0->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_texture0->setMagnificationFilter(QOpenGLTexture::Linear);
    m_texture0->setWrapMode(QOpenGLTexture::Repeat);
    m_texture0->bind(0);
    m_program->setUniformValue("texture0", 0);

    // Texture 1.
    const auto image1_path = src_path("../../../data/sunflowerField.jpg");
    const auto image1 = QImage{image1_path}.mirrored();
    m_texture1 = new QOpenGLTexture{image1};
    m_texture1->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_texture1->setMagnificationFilter(QOpenGLTexture::Linear);
    m_texture1->bind(1);
    m_program->setUniformValue("texture1", 1);
  }

  void initializeGL() override
  {
    glEnable(GL_DEPTH_TEST);

    initialize_shader_program();
    initialize_geometry();
    initialize_geometry_on_gpu();
    initialize_texture_on_gpu();

    // Initialize view matrix.
    m_view.setToIdentity();
    m_view.translate(0.f, 0.f, -10.f);

    // Bind the projection and view matrices once for all.
    m_program->bind();
    m_program->setUniformValue("view", m_view);

    // Bind the cube geometry once for all.
    m_vao->bind();
  }

  void paintGL() override
  {
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //m_program->bind();

    // Reset the projection matrix because the width and height can change.
    m_projection.setToIdentity();
    m_projection.perspective(45.f, float(width()) / height(), 0.1f, 100.f);
    m_program->setUniformValue("projection", m_projection);

    for (auto i = 0u; i < m_cubePositions.size(); ++i)
    {
      const auto& t = m_cubePositions[i];
      // Rotate and translate and the cube.
      auto transform = QMatrix4x4{};
      transform.setToIdentity();
      transform.translate(t.x(), t.y(), t.z());
      transform.rotate(std::pow(1.2f, (i + 1) * 5) * timer.elapsed_ms() / 10,
                       QVector3D{0.5f, 1.f, 0.f}.normalized());
      m_program->setUniformValue("transform", transform);

      // Draw triangles.
      glDrawArrays(GL_TRIANGLES, 0, m_vertices.size(0));
    }

    //m_program->release();
  }
};


int main(int argc, char **argv)
{
  QGuiApplication app(argc, argv);
  QSurfaceFormat format;
  format.setOption(QSurfaceFormat::DebugContext);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setVersion(3, 3);

  Window window;
  window.setFormat(format);
  window.resize(800, 600);
  window.show();

  QTimer timer;
  timer.start(20);
  QObject::connect(&timer, SIGNAL(timeout()), &window, SLOT(update()));

  return app.exec();
}
