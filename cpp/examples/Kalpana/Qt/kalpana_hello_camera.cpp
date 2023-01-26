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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Defines.hpp>
#include <DO/Sara/MultiViewGeometry/Datasets/Strecha.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/PinholeCamera.hpp>

#include <QGuiApplication>
#include <QKeyEvent>
#include <QOpenGLBuffer>
#include <QOpenGLDebugLogger>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWindow>
#include <QSurfaceFormat>
#include <QtCore/QException>
#include <QtCore/QObject>
#include <QtCore/QTimer>

#include <map>


using namespace DO::Sara;
using namespace std;
using namespace std::string_literals;


// Default camera values
// clang-format off
static const float YAW         = -90.0f;
static const float PITCH       =  0.0f;
static const float SPEED       =  1e-2f;
static const float SENSITIVITY =  1e-2f;
static const float ZOOM        =  45.0f;
// clang-format on


// The explorer's eye.
struct Camera
{
  Vector3f position{0.f, 0.f, 0.f};
  Vector3f front{0, 0, -1};
  Vector3f up{Vector3f::UnitY()};
  Vector3f right;
  Vector3f world_up{Vector3f::UnitY()};

  float yaw{YAW};
  float pitch{PITCH};
  float roll{0.f};

  float movement_speed{SPEED};
  float movement_sensitivity{SENSITIVITY};
  float zoom{ZOOM};

  Camera()
  {
    update();
  }

  auto move_left(float delta)
  {
    position -= movement_speed * delta * right;
  }

  auto move_right(float delta)
  {
    position += movement_speed * delta * right;
  }

  auto move_forward(float delta)
  {
    position += movement_speed * delta * front;
  }

  auto move_backward(float delta)
  {
    position -= movement_speed * delta * front;
  }

  auto move_up(float delta)
  {
    position += movement_speed * delta * up;
  }

  auto move_down(float delta)
  {
    position -= movement_speed * delta * up;
  }

  // pitch
  auto yes_head_movement(float delta)
  {
    pitch += movement_sensitivity * delta;
  }

  // yaw
  auto no_head_movement(float delta)
  {
    yaw += movement_sensitivity * delta;
  }

  auto maybe_head_movement(float delta)
  {
    roll += movement_sensitivity * delta;
  }

  auto update() -> void
  {
    Vector3f front1;

    // clang-format off
    front1 << cos(yaw * M_PI / 180) * cos(pitch * M_PI / 180.f),
              sin(pitch * M_PI / 180.f),
              sin(yaw * M_PI / 180.f) * cos(pitch * M_PI / 180.f);
    // clang-format on
    front = front1.normalized();

    right = front.cross(world_up).normalized();
    right =
        AngleAxisf(roll * float(M_PI) / 180, front).toRotationMatrix() * right;
    right.normalize();

    up = right.cross(front).normalized();
  }

  auto view_matrix() const -> QMatrix4x4
  {
    const auto qpos = QVector3D{position.x(), position.y(), position.z()};
    const auto qfront = QVector3D{front.x(), front.y(), front.z()};
    const auto qup = QVector3D{up.x(), up.y(), up.z()};

    auto view = QMatrix4x4{};
    view.lookAt(qpos, qpos + qfront, qup);

    return view;
  }
};


auto read_point_cloud(const std::string& h5_filepath) -> Tensor_<float, 2>
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};

  auto coords = MatrixXd{};
  h5_file.read_dataset("points", coords);
  // In OpenGL the y-axis is upwards. In the image coordinates, it is downwards.
  coords.row(1) *= -1;
  // In OpenGL, the cheiral constraint is actually a negative z.
  coords.row(2) *= -1;

  auto coords_tensorview =
      TensorView_<double, 2>{coords.data(), {coords.cols(), coords.rows()}};

  auto colors = Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);

  // Concatenate the data.
  auto vertex_data = Tensor_<double, 2>{{coords.cols(), 6}};
  vertex_data.matrix() << coords_tensorview.matrix(), colors.matrix();

  return vertex_data.cast<float>();
}

auto make_point_cloud(const std::string& h5_filepath)
{
  // Encode the vertex data in a tensor.
  const auto vertex_data = read_point_cloud(h5_filepath);
  SARA_DEBUG << "vertices =\n" << vertex_data.matrix().topRows(20) << std::endl;
  SARA_DEBUG << "min =\n"
             << vertex_data.matrix().colwise().minCoeff() << std::endl;
  SARA_DEBUG << "max =\n"
             << vertex_data.matrix().colwise().maxCoeff() << std::endl;
  return vertex_data;
}


class PointCloudObject : public QObject
{
public:
  PointCloudObject(const Tensor_<float, 2>& vertices, QObject* parent = nullptr)
    : QObject{parent}
    , m_vertices{vertices}
  {
    initialize_shader();
    initialize_geometry_on_gpu();
  }

  ~PointCloudObject()
  {
    m_program->release();

    m_vao->release();
    m_vao->destroy();

    m_vbo.release();
    m_vbo.destroy();
  }

  auto initialize_shader() -> void
  {
    SARA_DEBUG << "Initializing point cloud shader..." << std::endl;

    const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec3 out_color;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    gl_PointSize = 10.0;
    out_color = in_color;
  }
  )shader";

    const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
    if (dot(circCoord, circCoord) > 1.0)
        discard;

    float dist = length(gl_PointCoord - vec2(0.5));
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);

    frag_color = vec4(out_color, alpha);
  }
  )shader";

    m_program = new QOpenGLShaderProgram{parent()};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
    m_program->bind();
  }

  auto initialize_geometry_on_gpu() -> void
  {
    SARA_DEBUG << "Initializing point cloud data on GPU..." << std::endl;
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * static_cast<int>(sizeof(float));
    };

    const auto float_pointer = [](int offset) {
      return offset * static_cast<int>(sizeof(float));
    };

    m_vao->bind();

    // Copy the vertex data into the GPU buffer object.
    m_vbo.bind();
    m_vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vbo.allocate(m_vertices.data(),
                   static_cast<int>(m_vertices.size() * sizeof(float)));

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
    m_program->enableAttributeArray(arg_pos["in_color"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_color"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(3),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    m_vao->release();
  }

  auto render(const QMatrix4x4& projection,  //
              const QMatrix4x4& view,        //
              const QMatrix4x4& transform) -> void
  {
    m_program->bind();
    {
      m_program->setUniformValue("projection", projection);
      m_program->setUniformValue("view", view);
      m_program->setUniformValue("transform", transform);

      m_vao->bind();
      glDrawArrays(GL_POINTS, 0, m_vertices.size(0));
      m_vao->release();
    }
    m_program->release();
  }

private:
  //! @brief CPU data.
  Tensor_<float, 2> m_vertices;

  //! @brief GPU data.
  QOpenGLShaderProgram* m_program{nullptr};
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1}};
};


class CheckerBoardObject : public QObject
{
public:
  CheckerBoardObject(int rows_ = 20, int cols_ = 20, float scale = 10.f,
                     QObject* parent = nullptr)
    : QObject{parent}
  {
    initialize_geometry(rows_, cols_, scale);
    initialize_shader();
    initialize_geometry_on_gpu();
  }

  ~CheckerBoardObject()
  {
    m_vao->release();
    m_vao->destroy();

    m_vbo.release();
    m_vbo.destroy();

    m_ebo.release();
    m_ebo.destroy();
  }

  auto initialize_geometry(int rows_, int cols_, float scale) -> void
  {
    SARA_DEBUG << "Initialize geometry..." << std::endl;

    rows = rows_;
    cols = cols_;

    SARA_CHECK(rows);
    SARA_CHECK(cols);

    m_vertices = Tensor_<float, 2>{{4 * rows * cols, 6}};
    m_triangles = Tensor_<unsigned int, 2>{{2 * rows * cols, 3}};

    auto v_mat = m_vertices.matrix();
    auto t_mat = m_triangles.matrix();
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        const auto ij = cols * i + j;

        // clang-format off

        // Coordinates.
        v_mat.block(4 * ij, 0, 4, 3) << //
          // coords
          i +  0.5f, 0.0f, j +  0.5f,  // top-right
          i +  0.5f, 0.0f, j + -0.5f,  // bottom-right
          i + -0.5f, 0.0f, j + -0.5f,  // bottom-left
          i + -0.5f, 0.0f, j +  0.5f;  // top-left

        // Set colors.
        if (i % 2 == 0 && j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();
        else if (i % 2 == 0 && j % 2 == 1)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else if (i % 2 == 1 && j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else // (i % 2 == 1 and j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();

        t_mat.block(2 * ij, 0, 2, 3) <<
          4 * ij + 0, 4 * ij + 1, 4 * ij + 2,
          4 * ij + 2, 4 * ij + 3, 4 * ij + 0;

        // clang-format on
      }
    }

    // Translate.
    v_mat.col(0).array() -= rows / 2.f;
    v_mat.col(2).array() -= cols / 2.f;
    // Rescale.
    v_mat.leftCols(3) *= scale;
  }

  auto initialize_shader() -> void
  {
    SARA_DEBUG << "Initialize shader..." << std::endl;

    const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec3 out_color;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
  }
  )shader";

    const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    frag_color = vec4(out_color, 0.1);
  }
  )shader";

    m_program = new QOpenGLShaderProgram{parent()};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
  }

  auto initialize_geometry_on_gpu() -> void
  {
    SARA_DEBUG << "Initialize checkerboard geometry data on GPU..."
               << std::endl;
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    if (!m_ebo.create())
      throw QException{};

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * static_cast<int>(sizeof(float));
    };

    const auto float_pointer = [](int offset) {
      return offset * static_cast<int>(sizeof(float));
    };

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

    // Unbind the vbo to protect its data.
    m_vao->release();
  }

  auto render(const QMatrix4x4& projection,  //
              const QMatrix4x4& view,        //
              const QMatrix4x4& transform) -> void
  {
    m_program->bind();
    {
      m_program->setUniformValue("projection", projection);
      m_program->setUniformValue("view", view);
      m_program->setUniformValue("transform", transform);

      m_vao->bind();
      glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_triangles.size()),
                     GL_UNSIGNED_INT, 0);
      m_vao->release();
    }
    m_program->release();
  }

private:
  //! @{
  //! @brief CPU data.
  int rows;
  int cols;
  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  //! @}

  //! @{
  //! @brief GPU data.
  QOpenGLShaderProgram* m_program{nullptr};
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"out_color", 0}};
  //! @}
};


class ImagePlane : public QObject
{
public:
  ImagePlane(QObject* parent = nullptr)
    : QObject{parent}
  {
    initialize_shader();
  }

  ~ImagePlane()
  {
    m_program->release();

    m_vao->release();
    m_vao->destroy();

    m_vbo.release();
    m_vbo.destroy();

    m_ebo.release();
    m_ebo.destroy();
  }

  auto initialize_geometry() -> void
  {
    SARA_DEBUG << "Initialize image plane geometry..." << std::endl;

    const double w = m_image_sizes(0);
    const double h = m_image_sizes(1);

    // Encode the pixel coordinates of the image corners.
    auto corners = Matrix<double, 3, 4>{};
    // clang-format off
    corners <<
      w, w, 0, 0,
      0, h, h, 0,
      1, 1, 1, 1;
    // clang-format on
    SARA_DEBUG << "Pixel corners =\n" << corners << std::endl;

    SARA_DEBUG << "K =\n" << m_camera.K << std::endl;

    // Calculate the normalized camera coordinates.
    const Matrix3d K_inv = m_camera.K.inverse();
    corners = K_inv * corners;
    SARA_DEBUG << "Normalized corners =\n" << corners << std::endl;

    // Because the z is negative in OpenGL.
    corners.row(2) *= -1;
    // SARA_DEBUG << "Z-negative normalized corners =\n" << corners <<
    // std::endl;
    SARA_DEBUG << "Z-negative corners =\n" << corners << std::endl;

    // Vertex coordinates.
    m_vertices = Tensor_<float, 2>{{4, 5}};
    m_vertices.matrix().block<4, 3>(0, 0) = corners.transpose().cast<float>();

    // Texture coordinates.
    // clang-format off
    m_vertices.matrix().block<4, 2>(0, 3) <<
      1.0f, 0.0f,  // bottom-right
      1.0f, 1.0f,  // top-right
      0.0f, 1.0f,  // top-left
      0.0f, 0.0f;  // bottom-left
    // clang-format on

    SARA_DEBUG << "m_vertices =\n" << m_vertices.matrix() << std::endl;

    m_triangles = Tensor_<unsigned int, 2>{{2, 3}};
    // clang-format off
    m_triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
    // clang-format on
  }

  auto initialize_shader() -> void
  {
    SARA_DEBUG << "Initialize shader..." << std::endl;

    const auto vertex_shader_source = R"shader(
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_tex_coords;

  uniform mat4 transform;
  uniform mat4 view;
  uniform mat4 projection;

  out vec2 out_tex_coords;

  void main()
  {
    gl_Position = projection * view * transform * vec4(in_coords, 1.0);
    out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
  }
  )shader";

    const auto fragment_shader_source = R"shader(
#version 330 core
  in vec2 out_tex_coords;
  out vec4 frag_color;

  uniform sampler2D texture0;

  void main()
  {
    frag_color = texture(texture0, out_tex_coords);
    frag_color.w = 0.2;
  }
  )shader";

    m_program = new QOpenGLShaderProgram{parent()};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
  }

  auto initialize_geometry_on_gpu() -> void
  {
    SARA_DEBUG << "Initialize image plane geometry data on GPU..." << std::endl;
    m_vao = new QOpenGLVertexArrayObject{this};
    if (!m_vao->create())
      throw QException{};

    if (!m_vbo.create())
      throw QException{};

    if (!m_ebo.create())
      throw QException{};

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * static_cast<int>(sizeof(float));
    };

    const auto float_pointer = [](int offset) {
      return offset * static_cast<int>(sizeof(float));
    };

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
    m_program->setAttributeBuffer(/* location */ arg_pos["in_coords"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(0),
                                  /* tupleSize */ 3,
                                  /* stride */ row_bytes(m_vertices));

    // Specify that the vertex shader param 1 corresponds to the first 3 float
    // data of the buffer object.
    m_program->enableAttributeArray(arg_pos["in_tex_coords"]);
    m_program->setAttributeBuffer(/* location */ arg_pos["in_tex_coords"],
                                  /* GL_ENUM */ GL_FLOAT,
                                  /* offset */ float_pointer(3),
                                  /* tupleSize */ 2,
                                  /* stride */ row_bytes(m_vertices));

    // Unbind the vbo to protect its data.
    m_vao->release();
  }

  auto set_image(const string& image_path) -> void
  {
    const auto image = QImage{QString::fromStdString(image_path)}.mirrored();
    qDebug() << image.width() << " " << image.height();
    m_image_sizes << image.width(), image.height();
    m_texture = new QOpenGLTexture{image};
    m_texture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_texture->setMagnificationFilter(QOpenGLTexture::Linear);
  }

  auto set_camera(const PinholeCameraDecomposition& camera)
  {
    m_camera = camera;
    initialize_geometry();
    initialize_geometry_on_gpu();
  }

  auto render(const QMatrix4x4& projection,  //
              const QMatrix4x4& view,        //
              const QMatrix4x4& transform) -> void
  {
    m_program->bind();
    {
      m_program->setUniformValue("projection", projection);
      m_program->setUniformValue("view", view);
      m_program->setUniformValue("transform", transform);

      m_texture->bind(0);

      m_vao->bind();
      glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(m_triangles.size()),
                     GL_UNSIGNED_INT, 0);
      m_vao->release();
    }
    m_program->release();
  }

  auto destroy_texture()
  {
    if (m_texture == nullptr)
      return;

    m_texture->release();
    m_texture->destroy();
    delete m_texture;

    m_texture = nullptr;
  }

private:
  //! @{
  //! @brief CPU data.
  Vector2i m_image_sizes;
  PinholeCameraDecomposition m_camera{normalized_camera()};
  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  //! @}

  //! @{
  //! @brief GPU data.
  QOpenGLShaderProgram* m_program{nullptr};
  QOpenGLVertexArrayObject* m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};
  QOpenGLTexture* m_texture{nullptr};
  std::map<std::string, int> arg_pos = {{"in_coords", 0},      //
                                        {"in_tex_coords", 1},  //
                                        {"out_color", 0}};
  //! @}
};


class Window : public QOpenGLWindow
{
private:
  QString m_h5_file;
  QMatrix4x4 m_projection;
  QMatrix4x4 m_view;
  QMatrix4x4 m_transform;

  Camera m_camera;
  Timer m_timer;
  CheckerBoardObject* m_checkerboard{nullptr};
  PointCloudObject* m_pointCloud{nullptr};
  ImagePlane* m_imagePlane{nullptr};

public:
  Window(const QString& h5_file)
    : m_h5_file{h5_file}
  {
  }

  ~Window()
  {
    makeCurrent();
    {
      // TODO: refactor this kludge.
      if (m_imagePlane != nullptr)
        m_imagePlane->destroy_texture();
    }
    doneCurrent();
  }

  void initializeGL() override
  {
    // You absolutely need this for 3D objects!
    glEnable(GL_DEPTH_TEST);

    // Setup options for point cloud rendering.
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Instantiate the objects.
    m_checkerboard = new CheckerBoardObject{20, 20, 10, context()};
    m_pointCloud = new PointCloudObject{
        make_point_cloud(m_h5_file.toStdString()), context()};
#if 0
    m_imagePlane = new ImagePlane{context()};
#endif

    if (m_imagePlane != nullptr)
    {
      // Read HDF5 file.
      auto h5_file = H5File{m_h5_file.toStdString(), H5F_ACC_RDONLY};
      auto data_dir = std::string{};
      h5_file.read_dataset("dataset_folder", data_dir);

      auto image_filepath = std::string{};
      h5_file.read_dataset("image_1", image_filepath);
      m_imagePlane->set_image(image_filepath);

      auto K_filepath = std::string{};
      h5_file.read_dataset("K", K_filepath);
      auto camera = PinholeCameraDecomposition{
          read_internal_camera_parameters(K_filepath), Matrix3d::Identity(),
          Vector3d::Zero()};
      m_imagePlane->set_camera(camera);
    }
  }

  void paintGL() override
  {
    const qreal retinaScale = devicePixelRatio();
    glViewport(0, 0, width() * retinaScale, height() * retinaScale);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Reset the projection matrix.
    m_projection.setToIdentity();
    m_projection.perspective(45.f, float(width()) / height(), 0.1f, 1000.f);

    // Update the view matrix.
    m_view = m_camera.view_matrix();

    // Checkerboard.
    m_transform.setToIdentity();

    // m_transform.rotate(std::pow(1.5, 5) * m_timer.elapsed_ms() / 500,
    //                    QVector3D{0.5f, 1.0f, 0.0f}.normalized());

    // m_checkerboard->render(m_projection, m_view, m_transform);
    m_pointCloud->render(m_projection, m_view, m_transform);
    if (m_imagePlane != nullptr)
      m_imagePlane->render(m_projection, m_view, m_transform);
  }

protected:
  void keyPressEvent(QKeyEvent* ev) override
  {
    move_camera_from_keyboard(ev->key());
  }

  auto move_camera_from_keyboard(int key) -> void
  {
    const auto delta = 140.f;
    if (Qt::Key_W == key)
    {
      m_camera.move_forward(delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_S == key)
    {
      m_camera.move_backward(delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_A == key)
    {
      m_camera.move_left(delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_D == key)
    {
      m_camera.move_right(delta);
      m_camera.update();
      update();
    }

    if (Qt::Key_H == key)
    {
      m_camera.no_head_movement(-delta);  // CCW
      m_camera.update();
      update();
    }
    if (Qt::Key_L == key)
    {
      m_camera.no_head_movement(+delta);  // CW
      m_camera.update();
      update();
    }

    if (Qt::Key_K == key)
    {
      m_camera.yes_head_movement(+delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_J == key)
    {
      m_camera.yes_head_movement(-delta);
      m_camera.update();
      update();
    }

    if (Qt::Key_R == key)
    {
      m_camera.move_up(delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_F == key)
    {
      m_camera.move_down(delta);
      m_camera.update();
      update();
    }

    if (Qt::Key_U == key)
    {
      m_camera.maybe_head_movement(-delta);
      m_camera.update();
      update();
    }
    if (Qt::Key_I == key)
    {
      m_camera.maybe_head_movement(+delta);
      m_camera.update();
      update();
    }
  }
};


int main(int argc, char** argv)
{
  using namespace std::string_literals;
#ifdef __APPLE__
  auto geometry_h5_file = "/Users/david/Desktop/geometry.h5"s;
#else
  auto geometry_h5_file = "/home/david/Desktop/geometry.h5"s;
#endif
  if (argc >= 2)
    geometry_h5_file = argv[1];

  QGuiApplication app(argc, argv);
  QSurfaceFormat format;
  format.setOption(QSurfaceFormat::DebugContext);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setVersion(3, 3);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
  format.setSwapInterval(1);

  Window window{QString::fromStdString(geometry_h5_file)};
  window.setFormat(format);
  window.resize(800, 600);
  window.show();

  QTimer timer;
  timer.start(20);
  QObject::connect(&timer, SIGNAL(timeout()), &window, SLOT(update()));

  return app.exec();
}
