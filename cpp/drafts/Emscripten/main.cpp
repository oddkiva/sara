#include <drafts/OpenCL/GL.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>

#include <iostream>
#include <map>

#ifdef EMSCRIPTEN
#  include <emscripten/emscripten.h>
#  define GLFW_INCLUDE_ES3
#endif

#include <GLFW/glfw3.h>


using namespace DO::Sara;
using namespace std;


GLFWwindow* window;
int windowWidth;
int windowHeight;


static void window_size_callback(GLFWwindow* /* window */, int width,
                                 int height)
{
  std::cout << "window_size_callback received width: " << width
            << "  height: " << height << std::endl;
}

static void key_callback(GLFWwindow* /* window */, int key, int /* scancode */,
                         int action, int /* modifier */)
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


auto vertices = Tensor_<float, 2>{{3, 6}};
auto vao = GL::VertexArray{};
auto vbo = GL::Buffer{};


void render_frame()
{
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw triangles
  glBindVertexArray(vao); // geometry specified by the VAO.
  glDrawArrays(GL_POINTS, 0, vertices.size(0)); //

  glfwSwapBuffers(window);
  glfwPollEvents();
}

int main()
{
  if (glfwInit() != GL_TRUE)
  {
    std::cout << "Failed to initialize GLFW!" << std::endl;
    glfwTerminate();
    return 1;
  }

  window = glfwCreateWindow(512, 512, "OpenGL Window", NULL, NULL);
  if (!window)
  {
    std::cout << "Failed to create window!" << std::endl;
    glfwTerminate();
    return 1;
  }

  // Create a vertex shader.
  std::map<std::string, int> arg_pos = {{"in_coords", 0},  //
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
  auto vertex_shader = GL::Shader{};
  vertex_shader.create_from_source(GL_VERTEX_SHADER, vertex_shader_source);

  const auto fragment_shader_source =R"shader(#version 300 es
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
  auto fragment_shader = GL::Shader{};
  fragment_shader.create_from_source(GL_FRAGMENT_SHADER,
                                     fragment_shader_source);

  auto shader_program = GL::ShaderProgram{};
  shader_program.create();
  shader_program.attach(vertex_shader, fragment_shader);

  // clang-format off
  vertices.flat_array() <<
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

  vao.generate();
  vbo.generate();

  // Specify the vertex attributes here.
  {
    glBindVertexArray(vao);

    // Copy the vertex data into the GPU buffer object.
    vbo.bind_vertex_data(vertices);

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

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_DEPTH_TEST);

  // Activate the shader program once and for all.
  shader_program.use(true);

  glfwMakeContextCurrent(window);
  glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
  glfwSetKeyCallback(window, key_callback);
  glClearColor(0.0f, 1.0f, 1.0f, 1.0f);

#ifdef EMSCRIPTEN
  emscripten_set_main_loop(render_frame, 0, true);
#else
  while (!glfwWindowShouldClose(window))
    render_frame();
#endif

  vertex_shader.destroy();
  fragment_shader.destroy();
  vao.destroy();
  vbo.destroy();

  glfwTerminate();
  return EXIT_SUCCESS;
}
