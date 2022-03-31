#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>

#include <iostream>
#include <map>
#include <memory>

#ifdef EMSCRIPTEN
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include <GLFW/glfw3.h>


namespace sara = DO::Sara;

using namespace DO::Sara;
using namespace std;


struct MyGLFW
{
  static GLFWwindow* window;
  static int width;
  static int height;

  static auto initialize() -> bool
  {
    if (glfwInit() != GL_TRUE)
    {
      std::cout << "Failed to initialize GLFW!" << std::endl;
      glfwTerminate();
      return false;
    }

    window = glfwCreateWindow(512, 512, "OpenGL Window", NULL, NULL);
    if (!MyGLFW::window)
    {
      std::cout << "Failed to create window!" << std::endl;
      glfwTerminate();
      return false;
    }

    glfwMakeContextCurrent(window);

    // Set the appropriate mouse and keyboard callbacks.
    glfwGetFramebufferSize(window, &width, &height);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetKeyCallback(window, key_callback);

    return true;
  }

  static void window_size_callback(GLFWwindow* /* window */, int width,
                                   int height)
  {
    std::cout << "window_size_callback received width: " << width
              << "  height: " << height << std::endl;
  }

  static void key_callback(GLFWwindow* /* window */, int key,
                           int /* scancode */, int action, int /* modifier */)
  {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
      glfwSetWindowShouldClose(window, 1);

    if (key == GLFW_KEY_ENTER)
      std::cout << "Hit Enter!" << std::endl;
  }

  static void mouse_callback(GLFWwindow* /* window */, int button,
                             int /* action */, int /* modifiers */)
  {
    std::cout << "Clicked mouse button: " << button << "!" << std::endl;
  }
};

GLFWwindow* MyGLFW::window = nullptr;
int MyGLFW::width = -1;
int MyGLFW::height = -1;


struct Scene
{
  // Host geometry data
  Tensor_<float, 2> vertices;

  // OpenGL/Device geometry data.
  sara::GL::VertexArray vao;
  sara::GL::Buffer vbo;

  // OpenGL shaders.
  sara::GL::Shader vertex_shader;
  sara::GL::Shader fragment_shader;
  sara::GL::ShaderProgram shader_program;

  static std::unique_ptr<Scene> _scene;

  static auto instance() -> Scene&
  {
    if (_scene.get() == nullptr)
      throw std::runtime_error{"Please initialize the scene first"};
    return *_scene;
  }

  static auto initialize() -> void
  {
    if (_scene == nullptr)
      _scene.reset(new Scene);

    // Initialize the host CPU data.
    _scene->vertices.resize(3, 6);
    // clang-format off
    _scene->vertices.flat_array() <<
    // coords           color
      -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // left
       0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f,  // right
       0.0f,  0.5f, 0.0f, 0.0f, 0.0f, 1.0f;  // top
    // clang-format on

    const auto row_bytes = [](const TensorView_<float, 2>& data) {
      return data.size(1) * sizeof(float);
    };
    const auto float_pointer = [](int offset) {
      return reinterpret_cast<void*>(offset * sizeof(float));
    };

    // Create a vertex shader.
    const std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
                                                {"in_color", 1},   //
                                                {"out_color", 0}};

    const auto vertex_shader_source = R"shader(#version 300 es
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
    _scene->vertex_shader.create_from_source(GL_VERTEX_SHADER,
                                             vertex_shader_source);

    // Create a fragment shader.
    const auto fragment_shader_source = R"shader(#version 300 es
    precision mediump float;

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
    _scene->fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                               fragment_shader_source);

    // Create the whole shader program.
    _scene->shader_program.create();
    _scene->shader_program.attach(_scene->vertex_shader,
                                  _scene->fragment_shader);

    _scene->vao.generate();
    _scene->vbo.generate();

    // Specify the vertex attributes here.
    {
      glBindVertexArray(_scene->vao);

      // Copy the vertex data into the GPU buffer object.
      _scene->vbo.bind_vertex_data(_scene->vertices);

      // Specify that the vertex shader param 0 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos.at("in_coords"), 3 /* 3D points */, GL_FLOAT,
                            GL_FALSE, row_bytes(_scene->vertices),
                            float_pointer(0));
      glEnableVertexAttribArray(arg_pos.at("in_coords"));

      // Specify that the vertex shader param 1 corresponds to the first 3 float
      // data of the buffer object.
      glVertexAttribPointer(arg_pos.at("in_color"), 3 /* 3D colors */, GL_FLOAT,
                            GL_FALSE, row_bytes(_scene->vertices),
                            float_pointer(3));
      glEnableVertexAttribArray(arg_pos.at("in_color"));

      // Unbind the vbo to protect its data.
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);
    }
  }

  static auto destroy_opengl_data() -> void
  {
    _scene->vertex_shader.destroy();
    _scene->fragment_shader.destroy();
    _scene->vao.destroy();
    _scene->vbo.destroy();
  }

  static auto render_frame()
  {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw triangles
    const auto& scene = instance();
    glBindVertexArray(scene.vao);  // geometry specified by the VAO.
    glDrawArrays(GL_POINTS, 0, scene.vertices.size(0));  //

    glfwSwapBuffers(MyGLFW::window);
    glfwPollEvents();
  }
};

std::unique_ptr<Scene> Scene::_scene = nullptr;


int main()
{
  try
  {
    if (!MyGLFW::initialize())
      return EXIT_FAILURE;

    Scene::initialize();
    Scene::instance().shader_program.use(true);

    // Specific rendering options.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    // Initialize the background.
    glClearColor(0.0f, 1.0f, 1.0f, 1.0f);

#ifdef EMSCRIPTEN
    emscripten_set_main_loop(Scene::render_frame, 0, 1);
#else
    while (!glfwWindowShouldClose(MyGLFW::window))
      Scene::render_frame();
#endif

    Scene::destroy_opengl_data();

    glfwTerminate();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
