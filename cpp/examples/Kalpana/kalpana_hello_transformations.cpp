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

#include <DO/Kalpana/3D/OpenGLWindow.hpp>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <Eigen/Geometry>

#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QtCore/QException>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLTexture>
#include <QtGui/QOpenGLVertexArrayObject>

#include <map>


using namespace DO::Sara;
using namespace std;


class Window : public OpenGLWindow
{
private:
  QOpenGLShaderProgram* m_program{nullptr};

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"in_tex_coords", 2},   //
                                        {"out_color", 0}};

  const char* vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;
  layout (location = 2) in vec2 in_tex_coords;

  uniform mat4 transform;

  out vec3 out_color;
  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = transform * vec4(in_coords, 1.0);
    out_color = in_color;
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";

  const char* fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
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
    //frag_color = mix(texture(texture0, out_tex_coords),
    //                 texture(texture1, out_tex_coords), 0.5)
    //           * vec4(out_color, 1.0);
  }
  )shader";

  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};

  QOpenGLTexture texture0{QOpenGLTexture::Target2D};
  QOpenGLTexture texture1{QOpenGLTexture::Target2D};

  Timer timer;

public:
  Window() = default;

  ~Window()
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
    SARA_DEBUG << "Initialize shader program" << std::endl;
    m_program = new QOpenGLShaderProgram{this};
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

    // Encode the vertex data in a tensor.
    m_vertices = Tensor_<float, 2>{{4, 8}};
    m_vertices.flat_array() << //
      // coords            color              texture coords
       0.5f, -0.5f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,  // bottom-right
       0.5f,  0.5f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 1.0f,  // top-right
      -0.5f,  0.5f, 0.0f,  1.0f, 1.0f, 0.0f,  0.0f, 1.0f,  // top-left
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f;  // bottom-left

    m_triangles = Tensor_<unsigned int, 2>{{2, 3}};
    m_triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
  }

  void initialize_geometry_on_gpu()
  {
    SARA_DEBUG << "Initialize geometry data on GPU" << std::endl;
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    if (!m_ebo.create())
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

    // Copy geometry data.
    m_ebo.bind();
    m_ebo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_ebo.allocate(m_triangles.data(),
                   m_triangles.size() * sizeof(unsigned int));

    // Map the parameters to the argument position for the vertex shader.
    //
    // Vertex coordinates.
    m_program->enableAttributeArray(arg_pos["in_coords"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_coords"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(0),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    // Colors.
    m_program->enableAttributeArray(arg_pos["in_color"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_color"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(3),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    // Texture coordinates.
    m_program->enableAttributeArray(arg_pos["in_tex_coords"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_color"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(6),
                                  /* tupleSize */ 2,
                                  /* stride */ row_bytes(m_vertices));
  }

  void initialize_texture_on_gpu()
  {
    SARA_DEBUG << "Initialize texture data on GPU" << std::endl;

    // Texture 0.
    {
      const auto image_path = src_path("data/ksmall.jpg");
      SARA_DEBUG << image_path << endl;
      const auto image = QImage{QString{image_path}}.mirrored();

      texture0.setData(image);
      texture0.setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
      texture0.setMagnificationFilter(QOpenGLTexture::Linear);
      texture0.setWrapMode(QOpenGLTexture::Repeat);
    }

    // Texture 1.
    {
      const auto image_path = src_path("data/sunflowerField.jpg");
      SARA_DEBUG << image_path << endl;
      const auto image = QImage{QString{image_path}}.mirrored();
      texture1.setData(image);
      texture1.setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
      texture1.setMagnificationFilter(QOpenGLTexture::Linear);
    }

    // Specify that GL_TEXTURE0 is mapped to texture0 in the fragment shader
    // code.
    m_program->setUniformValue("texture0", 0);
    // Specify that GL_TEXTURE1 is mapped to texture1 in the fragment shader
    // code.
    m_program->setUniformValue("texture1", 1);
  }

  void initialize() override
  {
    initialize_shader_program();
    initialize_geometry();
    initialize_geometry_on_gpu();
    initialize_texture_on_gpu();
  }

  void render() override
  {
    SARA_DEBUG << "Rendering" << std::endl;

    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glActiveTexture(GL_TEXTURE1);

    texture0.bind(GL_TEXTURE0);
    texture1.bind(GL_TEXTURE1);
    SARA_DEBUG << "OK" << std::endl;

    auto transform = Transform<float, 3, Eigen::Projective>{};
    transform.setIdentity();
    transform.rotate(AngleAxisf(timer.elapsed_ms() / 1000, Vector3f::UnitZ()));
    transform.translate(Vector3f{0.25f, 0.25f, 0.f});

    const GLfloat *transform_data = transform.matrix().data();
    SARA_DEBUG << "OK" << std::endl;

    m_program->setUniformValueArray("transform", transform_data, 4, 4);
    SARA_DEBUG << "OK" << std::endl;

    //// Draw triangles.
    //m_vao->bind();
    //glDrawElements(GL_TRIANGLES, m_triangles.size(), GL_UNSIGNED_INT, 0);
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

  window.setAnimating(true);

  return app.exec();
}
