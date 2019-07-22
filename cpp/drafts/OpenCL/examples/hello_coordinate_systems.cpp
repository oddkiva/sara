#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
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

auto init_gl_boilerplate()
{
  // Initialize the windows manager.
  if (!glfwInit())
    throw std::runtime_error{"Error: failed to initialize GLFW!"};

#ifndef __APPLE__
  // Initialize GLEW.
  auto err = glewInit();
  if (err != GLEW_OK)
  {
    std::cerr << format("Error: failed to initialize GLEW: %s",
                        glewGetErrorString(err))
              << std::endl;
    return EXIT_FAILURE;
  }
#endif

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
}

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

Tensor_<float, 2> read_point_cloud(const std::string& h5_filepath)
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};
  auto coords = Tensor_<float, 2>{};
  h5_file.read_dataset("points", coords);
  return coords;
}

auto make_point_cloud()
{
  // Encode the vertex data in a tensor.
  auto coords = read_point_cloud("/Users/david/Desktop/geometry.h5");
  auto vertices = Tensor_<float, 2>{{coords.size(0), 5}};
  vertices.flat_array().fill(1.f);
  vertices.matrix().leftCols(3) = coords.matrix();
  vertices.matrix().leftCols(3) *= -1.f;

  SARA_DEBUG << "coords sizes = " << coords.sizes().transpose() << std::endl;
  SARA_DEBUG << "coords =\n" << coords.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "vertices =\n" << vertices.matrix().topRows(10) << std::endl;

  return vertices;
}

int main()
{
  init_gl_boilerplate();

  // Create a window.
  const auto width = 800;
  const auto height = 600;
  auto window =
      glfwCreateWindow(width, height, "Hello Coordinate Systems", nullptr, nullptr);
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer);

  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                        {"in_tex_coords", 1},   //
                                        {"out_color", 0}};

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
  auto vertex_shader = GL::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);


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
  auto fragment_shader = GL::Shader{};
  fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                     fragment_shader_source);

  auto shader_program = GL::ShaderProgram{};
  shader_program.create();
  shader_program.attach(vertex_shader, fragment_shader);

  vertex_shader.destroy();
  fragment_shader.destroy();

  // Encode the vertex data in a tensor.
  auto vertices = make_cube();
  //auto vertices = make_point_cloud();

  Vector3f cubePositions[] = {
      Vector3f(0.0f, 0.0f, 0.0f),    Vector3f(2.0f, 5.0f, -15.0f),
      Vector3f(-1.5f, -2.2f, -2.5f), Vector3f(-3.8f, -2.0f, -12.3f),
      Vector3f(2.4f, -0.4f, -3.5f),  Vector3f(-1.7f, 3.0f, -7.5f),
      Vector3f(1.3f, -2.0f, -2.5f),  Vector3f(1.5f, 2.0f, -2.5f),
      Vector3f(1.5f, 0.2f, -1.5f),   Vector3f(-1.3f, 1.0f, -1.5f)};

  const auto row_bytes = [](const TensorView_<float, 2>& data) {
    return data.size(1) * sizeof(float);
  };
  const auto float_pointer = [](int offset) {
    return reinterpret_cast<void*>(offset * sizeof(float));
  };

  auto vao = GL::VertexArray{};
  vao.generate();

  // Vertex attributes.
  auto vbo = GL::Buffer{};
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
    glVertexAttribPointer(arg_pos["in_tex_coords"], 2 /* texture coords */, GL_FLOAT,
                          GL_FALSE, row_bytes(vertices), float_pointer(3));
    glEnableVertexAttribArray(arg_pos["in_tex_coords"]);
  }

  // Texture data.
  auto texture0 = GL::Texture2D{};
  {
    // Read the image from the disk.
    auto image =
        imread<Rgb8>("/Users/david/GitLab/DO-CV/sara/data/ksmall.jpg");
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE0);
    texture0.setup_with_pretty_defaults(image);
  }

  auto texture1 = GL::Texture2D{};
  {
    // Read the image from the disk.
    auto image =
        imread<Rgb8>("/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg");
    // Flip vertically so that the image data matches OpenGL image coordinate
    // system.
    flip_vertically(image);

    // Copy the image to the GPU texture.
    glActiveTexture(GL_TEXTURE1);
    texture1.setup_with_pretty_defaults(image);
  }

  shader_program.use(true);
  // Specify that GL_TEXTURE0 is mapped to texture0 in the fragment shader code.
  shader_program.set_uniform_param("texture0", 0);
  // Specify that GL_TEXTURE1 is mapped to texture1 in the fragment shader code.
  shader_program.set_uniform_param("texture1", 1);

  auto timer = Timer{};

  // For point cloud rendering.
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  // You need this for 3D objects!
  glEnable(GL_DEPTH_TEST);

  // Display image.
  glfwSwapInterval(1);
  while (!glfwWindowShouldClose(window))
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    // Important.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture1);

    auto view = Transform<float, 3, Eigen::Projective>{};
    view.setIdentity();
    view.translate(Vector3f{0.f, 0.f, -10.f});
    shader_program.set_uniform_matrix4f("view", view.matrix().data());

    const Matrix4f projection =
        kalpana::perspective(45., 800. / 600., .1, 100.).cast<float>();
    shader_program.set_uniform_matrix4f("projection",
                                        projection.data());

    // Draw triangles.
    glBindVertexArray(vao);
    for (int i = 0; i < 10; i++)
    {
      auto transform = Transform<float, 3, Eigen::Projective>{};
      transform.setIdentity();
      transform.translate(cubePositions[i]);
      transform.rotate(AngleAxisf(std::pow(1.2, (i + 1) * 2) * timer.elapsed_ms() / 10000,
                                  Vector3f{0.5f, 1.0f, 0.0f}.normalized()));
      shader_program.set_uniform_matrix4f("transform",
                                          transform.matrix().data());

      glDrawArrays(GL_TRIANGLES, 0, vertices.size(0));
    }


    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  vao.destroy();
  vbo.destroy();

  // Clean up resources.
  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
