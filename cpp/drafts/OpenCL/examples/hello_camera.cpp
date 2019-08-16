#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Flip.hpp>

#include <DO/Kalpana/Math/Projection.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include <GLFW/glfw3.h>

#include <Eigen/Geometry>

#include <map>


using namespace DO::Sara;
using namespace std;

namespace kalpana = DO::Kalpana;


auto resize_framebuffer(GLFWwindow*, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

inline auto init_glfw_boilerplate()
{
  // Initialize the windows manager.
  if (!glfwInit())
    throw std::runtime_error{"Error: failed to initialize GLFW!"};

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

}

inline auto init_glew_boilerplate()
{
#ifndef __APPLE__
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << format("Error: failed to initialize GLEW: %s",
                        glewGetErrorString(err))
              << std::endl;
  }
#endif
}


// Default camera values
static const float YAW         = -90.0f;
static const float PITCH       =  0.0f;
static const float SPEED       =  1e-1f;
static const float SENSITIVITY =  1e-1f;
static const float ZOOM        =  45.0f;

// The explorer's eye.
struct Camera
{
  Vector3f position{10.f * Vector3f::UnitY()};
  Vector3f front{-Vector3f::UnitZ()};
  Vector3f up{Vector3f::UnitY()}; Vector3f right;
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

  auto view_matrix() -> Matrix4f
  {
    return look_at(position, position + front, up);
  }

  auto look_at(const Vector3f& eye, const Vector3f& center, const Vector3f& up)
      -> Matrix4f
  {
    Vector3f f = (center - eye).normalized();
    Vector3f u = up.normalized();
    Vector3f s = f.cross(u).normalized();
    u = s.cross(f);

    Matrix4f res;
    res <<
       s.x(),  s.y(),  s.z(), -s.dot(eye),
       u.x(),  u.y(),  u.z(), -u.dot(eye),
      -f.x(), -f.y(), -f.z(),  f.dot(eye),
           0,      0,      0,           1;

    return res;
    }
};


struct Time
{
  void update()
  {
    last_frame = current_frame;
    current_frame = timer.elapsed_ms();
    delta_time = current_frame - last_frame;
  }

  Timer timer;
  float delta_time = 0.f;
  float last_frame = 0.f;
  float current_frame = 0.f;
};

auto move_camera_from_keyboard(GLFWwindow* window, Camera& camera, Time& time)
{
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.move_forward(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.move_backward(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.move_left(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.move_right(time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera.no_head_movement(-time.delta_time); // CCW
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera.no_head_movement(+time.delta_time); // CW

  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    camera.yes_head_movement(+time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    camera.yes_head_movement(-time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
    camera.move_up(time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
    camera.move_down(time.delta_time);

  if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    camera.maybe_head_movement(-time.delta_time);
  if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
    camera.maybe_head_movement(+time.delta_time);

  camera.update();
}


auto read_point_cloud(const std::string& h5_filepath) -> Tensor_<float, 2>
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};

  auto coords = Tensor_<double, 2>{};
  h5_file.read_dataset("points", coords);
  coords.matrix() *= -1;

  auto colors = Tensor_<double, 2>{};
  h5_file.read_dataset("colors", colors);
  SARA_DEBUG << "OK" << std::endl;

  // Concatenate the data.
  auto vertex_data = Tensor_<double, 2>{{coords.size(0), 6}};
  vertex_data.matrix() << coords.matrix(), colors.matrix();

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
  SARA_DEBUG << "vertices =\n" << vertex_data.matrix() << std::endl;
  return vertex_data;
}

struct PointCloudObject
{
  PointCloudObject(const Tensor_<float, 2>& vertices_)
    : vertices{vertices_}
  {
    shader_program = make_shader();

    // =========================================================================
    // Encode the vertex data in a tensor.
    //
    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    vao.generate();


    // =========================================================================
    // Setup Vertex attributes on the GPU side.
    //
    vbo.generate();
    {
      glBindVertexArray(vao);

      // Copy vertex data.
      vbo.bind_vertex_data(vertices);

      // Map the parameters to the argument position for the vertex shader.
      //
      // Vertex coordinates.
      glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(0));
      glEnableVertexAttribArray(arg_pos["in_coords"]);

      // Texture coordinates.
      glVertexAttribPointer(arg_pos["in_color"], 3 /* RGB_COLOR */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(3));
      glEnableVertexAttribArray(arg_pos["in_color"]);
    }
  }

  auto make_shader() -> GL::ShaderProgram
  {
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
    auto vertex_shader = GL::Shader{};
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


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
    auto fragment_shader = GL::Shader{};
    fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                       fragment_shader_source);

    auto shader_program = GL::ShaderProgram{};
    shader_program.create();
    shader_program.attach(vertex_shader, fragment_shader);

    vertex_shader.destroy();
    fragment_shader.destroy();

    return shader_program;
  }

  void destroy()
  {
    vao.destroy();
    vbo.destroy();
  }

  Tensor_<float, 2> vertices;
  GL::Buffer vbo;
  GL::VertexArray vao;
  GL::ShaderProgram shader_program;
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1}};
};


struct CheckerBoardObject
{
  CheckerBoardObject(int rows_ = 20, int cols_ = 20, float scale = 10.f)
    : rows{rows_}
    , cols{cols_}
  {
    shader_program = make_shader();

    vertices = Tensor_<float, 2>{{4 * rows * cols, 6}};
    triangles = Tensor_<unsigned int, 2>{{2 * rows * cols, 3}};

    auto v_mat = vertices.matrix();
    auto t_mat = triangles.matrix();
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

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    vao.generate();
    vbo.generate();
    ebo.generate();

    // Specify the vertex attributes here.
    {
      glBindVertexArray(vao);

      // Copy the vertex data into the GPU buffer object.
      vbo.bind_vertex_data(vertices);

      // Copy the triangles data into the GPU buffer object.
      ebo.bind_triangles_data(triangles);

      // Specify that the vertex shader param 0 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos["in_coords"], 3 /* 3D points */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(0));
      glEnableVertexAttribArray(arg_pos["in_coords"]);

      // Specify that the vertex shader param 1 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos["in_color"], 3 /* 3D colors */, GL_FLOAT,
                            GL_FALSE, row_bytes(vertices), float_pointer(3));
      glEnableVertexAttribArray(arg_pos["in_color"]);

      // Unbind the vbo to protect its data.
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);
    }
  }

  auto make_shader() -> GL::ShaderProgram
  {
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
    auto vertex_shader = GL::Shader{};
    vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


    const auto fragment_shader_source = R"shader(
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    frag_color = vec4(out_color, 0.1);
  }
  )shader";
    auto fragment_shader = GL::Shader{};
    fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                       fragment_shader_source);

    auto shader_program = GL::ShaderProgram{};
    shader_program.create();
    shader_program.attach(vertex_shader, fragment_shader);

    vertex_shader.destroy();
    fragment_shader.destroy();

    return shader_program;
  }

  void destroy()
  {
    vao.destroy();
    vbo.destroy();
    ebo.destroy();
  }

  int rows{100};
  int cols{100};
  Tensor_<float, 2> vertices;
  Tensor_<unsigned int, 2> triangles;
  GL::Buffer vbo;
  GL::Buffer ebo;
  GL::VertexArray vao;
  GL::ShaderProgram shader_program;
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_color", 1},   //
                                        {"out_color", 0}};
};


int main()
{
  // ==========================================================================
  // Boilerplate code for display initialization.
  //
  init_glfw_boilerplate();

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window =
      glfwCreateWindow(width, height, "Hello Point Cloud", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer);
  //glfwSetCursorPosCallback(window, move_camera_from_mouse);

  init_glew_boilerplate();


  // ==========================================================================
  // Create objects to display.
  //
  auto point_cloud_object = PointCloudObject{make_point_cloud()};
  auto checkerboard = CheckerBoardObject{};
  auto camera = Camera{};


  // ==========================================================================
  // Setup options for point cloud rendering.
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  // You absolutely need this for 3D objects!
  glEnable(GL_DEPTH_TEST);

  auto time = Time{};

  // Initialize the projection matrix once for all.
  const Matrix4f projection =
      kalpana::perspective(45., 800. / 600., .1, 1000.).cast<float>();


  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    // Calculate the elapsed time.
    time.update();

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);

    // Camera interaction with keyboard.
    move_camera_from_keyboard(window, camera, time);
    auto view_matrix = camera.view_matrix();

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    // Important.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Transform matrix.
    auto transform = Transform<float, 3, Eigen::Projective>{};
    transform.setIdentity();

    // Draw the checkerboard.
    checkerboard.shader_program.use();
    checkerboard.shader_program.set_uniform_matrix4f(
        "transform", transform.matrix().data());
    checkerboard.shader_program.set_uniform_matrix4f("view",
                                                     view_matrix.data());
    checkerboard.shader_program.set_uniform_matrix4f("projection",
                                                     projection.data());
    glBindVertexArray(checkerboard.vao);
    glDrawElements(GL_TRIANGLES, checkerboard.triangles.size(), GL_UNSIGNED_INT, 0);


    // Rotate the point cloud.
    transform.rotate(AngleAxisf(std::pow(1.5, 5) * time.last_frame / 10000,
                                Vector3f{0.5f, 1.0f, 0.0f}.normalized()));

    // Draw point cloud.
    point_cloud_object.shader_program.use();
    point_cloud_object.shader_program.set_uniform_matrix4f(
        "transform", transform.matrix().data());
    point_cloud_object.shader_program.set_uniform_matrix4f("view",
                                                           view_matrix.data());
    point_cloud_object.shader_program.set_uniform_matrix4f("projection",
                                                           projection.data());

    glBindVertexArray(point_cloud_object.vao);
    glDrawArrays(GL_POINTS, 0, point_cloud_object.vertices.size(0));


    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  point_cloud_object.destroy();
  checkerboard.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
