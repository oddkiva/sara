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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Defines.hpp>

#include <QGuiApplication>
#include <QSurfaceFormat>
#include <QtCore/QException>
#include <QtGui/QKeyEvent>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLTexture>
#include <QtGui/QOpenGLVertexArrayObject>

#include <map>


using namespace DO::Sara;
using namespace std;


// Default camera values
static const float YAW         = -90.0f;
static const float PITCH       =  0.0f;
static const float SPEED       =  1e-2f;
static const float SENSITIVITY =  1e-2f;
static const float ZOOM        =  45.0f;


// The explorer's eye.
struct Camera
{
  Vector3f position{0.f, 10.f, 10.f};
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

  auto update()
  {
    Vector3f front1;

    front1 << cos(yaw * M_PI / 180) * cos(pitch * M_PI / 180.f),
              sin(pitch * M_PI / 180.f),
              sin(yaw * M_PI / 180.f) * cos(pitch * M_PI / 180.f);
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
  coords.matrix() *= -1;
  auto coords_tensorview =
      TensorView_<double, 2>{coords.data(), {coords.cols(), coords.rows()}};

  auto colors = Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);

  // Concatenate the data.
  auto vertex_data = Tensor_<double, 2>{{coords.cols(), 6}};
  vertex_data.matrix() << coords_tensorview.matrix(), colors.matrix();

  return vertex_data.cast<float>();
}

auto make_point_cloud()
{
  // Encode the vertex data in a tensor.
#ifdef __APPLE__
  const auto vertex_data = read_point_cloud("/Users/david/Desktop/geometry.h5");
#else
  const auto vertex_data = read_point_cloud("/home/david/Desktop/geometry.h5");
#endif
  SARA_DEBUG << "vertices =\n" << vertex_data.matrix().topRows(20) << std::endl;
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
    gl_PointSize = 5.0;
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

    m_program = new QOpenGLShaderProgram{this};
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
  QOpenGLShaderProgram *m_program{nullptr};
  QOpenGLVertexArrayObject *m_vao{nullptr};
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

        // Coordinates.
        v_mat.block(4 * ij, 0, 4, 3) << //
          // coords
          i +  0.5f, 0.0f, j +  0.5f,  // top-right
          i +  0.5f, 0.0f, j + -0.5f,  // bottom-right
          i + -0.5f, 0.0f, j + -0.5f,  // bottom-left
          i + -0.5f, 0.0f, j +  0.5f;  // top-left

        // Set colors.
        if (i % 2 == 0 and j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();
        else if (i % 2 == 0 and j % 2 == 1)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else if (i % 2 == 1 and j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setOnes();
        else // (i % 2 == 1 and j % 2 == 0)
          v_mat.block(4 * ij, 3, 4, 3).setZero();

        t_mat.block(2 * ij, 0, 2, 3) <<
          4 * ij + 0, 4 * ij + 1, 4 * ij + 2,
          4 * ij + 2, 4 * ij + 3, 4 * ij + 0;
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

    m_program = new QOpenGLShaderProgram{this};
    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                       vertex_shader_source);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                       fragment_shader_source);
    m_program->link();
  }

  auto initialize_geometry_on_gpu() -> void
  {
    SARA_DEBUG << "Initialize checkerboard geometry data on GPU..." << std::endl;
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

    // Copy the triangles data into the GPU buffer object.
    m_ebo.bind();
    m_ebo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_ebo.allocate(m_triangles.data(),
                   m_triangles.size() * sizeof(unsigned int));

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
      glDrawElements(GL_TRIANGLES, m_triangles.size(), GL_UNSIGNED_INT, 0);
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
  QOpenGLShaderProgram *m_program{nullptr};
  QOpenGLVertexArrayObject *m_vao{nullptr};
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
    initialize_geometry();
    initialize_shader();
    initialize_geometry_on_gpu();
  }

  ~ImagePlane()
  {
    m_vao->release();
    m_vao->destroy();

    m_vbo.release();
    m_vbo.destroy();

    m_ebo.release();
    m_ebo.destroy();

    m_texture->release();
    m_texture->destroy();
    delete m_texture;
  }

  auto initialize_geometry() -> void
  {
    SARA_DEBUG << "Initialize geometry..." << std::endl;

    m_vertices = Tensor_<float, 2>{{4, 5}};
    m_vertices.flat_array() << //
      // coords            texture coords
       0.5f, -0.5f, 0.0f,  1.0f, 0.0f,  // bottom-right
       0.5f,  0.5f, 0.0f,  1.0f, 1.0f,  // top-right
      -0.5f,  0.5f, 0.0f,  0.0f, 1.0f,  // top-left
      -0.5f, -0.5f, 0.0f,  0.0f, 0.0f;  // bottom-left

    m_triangles = Tensor_<unsigned int, 2>{{2, 3}};
    m_triangles.flat_array() <<
      0, 1, 2,
      2, 3, 0;
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
    frag_color.w = 0.15;
  }
  )shader";

    m_program = new QOpenGLShaderProgram{this};
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

    // Copy the triangles data into the GPU buffer object.
    m_ebo.bind();
    m_ebo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_ebo.allocate(m_triangles.data(),
                   m_triangles.size() * sizeof(unsigned int));

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

  auto set_image(const string& image_path =
                     src_path("../../../data/sunflowerField.jpg")) -> void
  {
    const auto image = QImage{QString::fromStdString(image_path)}.mirrored();
    m_image_sizes << image.width(), image.height();

    SARA_DEBUG << image.width() << " " << image.height() << std::endl;
    m_texture = new QOpenGLTexture{image};
    m_texture->setMinificationFilter(QOpenGLTexture::LinearMipMapLinear);
    m_texture->setMagnificationFilter(QOpenGLTexture::Linear);
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
      glDrawElements(GL_TRIANGLES, m_triangles.size(), GL_UNSIGNED_INT, 0);
      m_vao->release();
    }
    m_program->release();
  }

private:
  //! @{
  //! @brief CPU data.
  Vector2i m_image_sizes;
  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
  //! @}

  //! @{
  //! @brief GPU data.
  QOpenGLShaderProgram *m_program{nullptr};
  QOpenGLVertexArrayObject *m_vao{nullptr};
  QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
  QOpenGLBuffer m_ebo{QOpenGLBuffer::IndexBuffer};
  QOpenGLTexture *m_texture{nullptr};
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_tex_coords", 1},   //
                                        {"out_color", 0}};
  //! @}
};


class Window : public OpenGLWindow
{
private:
  QMatrix4x4 m_projection;
  QMatrix4x4 m_view;
  QMatrix4x4 m_transform;

  Camera m_camera;
  Timer m_timer;
  CheckerBoardObject *m_checkerboard{nullptr};
  PointCloudObject *m_pointCloud{nullptr};
  ImagePlane *m_imagePlane{nullptr};

public:
  Window() = default;

  ~Window()
  {
    m_context->makeCurrent(this);
    {
      //
    }
    m_context->doneCurrent();
  }

  void initialize() override
  {
    // Setup options for point cloud rendering.
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // You absolutely need this for 3D objects!
    glEnable(GL_DEPTH_TEST);

    // Instantiate the objects.
    m_checkerboard = new CheckerBoardObject{20, 20, 10, this};
    m_pointCloud = new PointCloudObject{make_point_cloud(), this};
    m_imagePlane = new ImagePlane{this};

#ifdef __APPLE__
    const auto data_dir =
        std::string{"/Users/david/Desktop/Datasets/sfm/castle_int"};
#else
    const auto data_dir =
        std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
#endif
    const auto file1 = "0000.png";
    m_imagePlane->set_image(data_dir + "/" + file1);
  }

  void render() override
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
    //m_checkerboard->render(m_projection, m_view, m_transform);

    m_imagePlane->render(m_projection, m_view, m_transform);

    //m_transform.rotate(std::pow(1.5, 5) * m_timer.elapsed_ms() / 500,
    //                   QVector3D{0.5f, 1.0f, 0.0f}.normalized());
    m_pointCloud->render(m_projection, m_view, m_transform);
  }

protected:
  void keyPressEvent(QKeyEvent *ev) override
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
      renderLater();
    }
    if (Qt::Key_S == key)
    {
      m_camera.move_backward(delta);
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_A == key)
    {
      m_camera.move_left(delta);
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_D == key)
    {
      m_camera.move_right(delta);
      m_camera.update();
      renderLater();
    }

    if (Qt::Key_Delete == key)
    {
      m_camera.no_head_movement(-delta);  // CCW
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_PageDown == key)
    {
      m_camera.no_head_movement(+delta);  // CW
      m_camera.update();
      renderLater();
    }

    if (Qt::Key_Home == key)
    {
      m_camera.yes_head_movement(+delta);
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_End == key)
    {
      m_camera.yes_head_movement(-delta);
      m_camera.update();
      renderLater();
    }

    if (Qt::Key_R == key)
    {
      m_camera.move_up(delta);
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_F == key)
    {
      m_camera.move_down(delta);
      m_camera.update();
      renderLater();
    }

    if (Qt::Key_Insert == key)
    {
      m_camera.maybe_head_movement(-delta);
      m_camera.update();
      renderLater();
    }
    if (Qt::Key_PageUp == key)
    {
      m_camera.maybe_head_movement(+delta);
      m_camera.update();
      renderLater();
    }
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
  window.setAnimating(true);

  return app.exec();
}
