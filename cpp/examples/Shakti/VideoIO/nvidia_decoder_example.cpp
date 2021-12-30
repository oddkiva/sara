#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cudaGL.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <DO/Shakti/Cuda/VideoIO.hpp>

#include "nvidia-video-codec-sdk-9.1.23/Utils/NvCodecUtils.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


//! @brief OpenGL stuff.
//! @{
struct PixelBuffer
{
  GLuint _pbo;

  auto allocate(int w, int h) -> void
  {
    glGenBuffersARB(1, &_pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, _pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * 4, NULL,
                    GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  }

  auto bind() const -> void
  {
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, _pbo);
  }

  auto release() -> void
  {
    glDeleteBuffersARB(1, &_pbo);
  }
};

struct Texture
{
  int width;
  int height;
  GLuint _tex;

  auto allocate(int w, int h) -> void
  {
    width = w;
    height = h;

    glGenTextures(1, &_tex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER,
                    GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER,
                    GL_NEAREST);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
  }

  auto release() -> void
  {
    glDeleteTextures(1, &_tex);
    width = 0;
    height = 0;
  }

  auto bind() const
  {
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, _tex);
  }

  auto display() -> void
  {
    glBegin(GL_QUADS);
    glTexCoord2f(0, static_cast<GLfloat>(height));
    glVertex2f(0, 0);
    glTexCoord2f(static_cast<GLfloat>(width), static_cast<GLfloat>(height));
    glVertex2f(1, 0);
    glTexCoord2f(static_cast<GLfloat>(width), 0);
    glVertex2f(1, 1);
    glTexCoord2f(0, 0);
    glVertex2f(0, 1);
    glEnd();
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
  }
};

struct Shader
{
  GLuint _shader;

  auto initialize() -> void
  {
    static const char* code =
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], RECT; \n"
        "END";
    glGenProgramsARB(1, &_shader);
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, _shader);
    glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
                       (GLsizei) strlen(code), (GLubyte*) code);
  }

  auto enable() -> void
  {
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, _shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
  }

  auto disable()
  {
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
  }

  auto release() -> void
  {
    glDeleteProgramsARB(1, &_shader);
  }
};
//! @}


//! @brief CUDA/OpenGL interoperability.
//! @{
struct CudaGraphicsResource
{
  CUgraphicsResource cuda_resource;
  CUdeviceptr device_data_ptr;
  size_t device_data_size;

  CudaGraphicsResource(const PixelBuffer& pbo)
  {
    ck(cuGraphicsGLRegisterBuffer(&cuda_resource, pbo._pbo,
                                  CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
    ck(cuGraphicsMapResources(1, &cuda_resource, 0));
    ck(cuGraphicsResourceGetMappedPointer(&device_data_ptr, &device_data_size,
                                          cuda_resource));
  }

  ~CudaGraphicsResource()
  {
    ck(cuGraphicsUnmapResources(1, &cuda_resource, 0));
    ck(cuGraphicsUnregisterResource(cuda_resource));
  }
};

auto cuda_async_copy(const DriverApi::DeviceBgraBuffer& src, PixelBuffer& dst)
    -> void
{
  CUDA_MEMCPY2D m{};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;

  const auto cuda_resource = CudaGraphicsResource{dst};

  // Source is dpFrame into which Decode() function writes data of individual
  // frame present in BGRA32 format.
  m.srcDevice = src.data;
  m.srcPitch = src.width * 4;

  // Destination is OpenGL buffer object mapped as a CUDA resource.
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = cuda_resource.device_data_ptr;
  m.dstPitch = cuda_resource.device_data_size / src.height;
  m.WidthInBytes = src.width * 4;
  m.Height = src.height;

  // Asynchronous copy from 2D device buffer to OpenGL buffer.
  ck(cuMemcpy2DAsync(&m, 0));
}
//! @}


int test_with_glfw(int argc, char** argv)
{
  if (argc < 2)
    return 1;

  // Initialize CUDA driver.
  DriverApi::init();

  // Create a CUDA context so that we can use the GPU device.
  const auto gpu_id = 0;
  auto cuda_context = DriverApi::CudaContext{gpu_id};
  cuda_context.make_current();

  const auto video_filepath = argv[1];

  // Initialize a CUDA-powered video streamer object.
  auto video_stream = shakti::VideoStream{video_filepath, cuda_context};

  // Create a video frame buffer.
  DriverApi::DeviceBgraBuffer device_bgra_buffer{video_stream.width(),
                                                 video_stream.height()};

  // Initialize GLFW.
  if (!glfwInit())
    throw std::runtime_error{"Failed to initialize GLFW!"};

  // Open a window on which GLFW can operate.
  const auto w = video_stream.width();
  const auto h = video_stream.height();
  auto window = glfwCreateWindow(w, h, "Hello World", nullptr, nullptr);
  if (window == nullptr)
    throw std::runtime_error{"Failed to create GLFW window!"};
  glfwMakeContextCurrent(window);

  // Initialize GLEW.
  glewInit();

  // Initialize OpenGL resources.
  auto pbo = PixelBuffer{};
  auto texture = Texture{};
  auto shader = Shader{};
  pbo.allocate(w, h);
  texture.allocate(w, h);
  shader.initialize();

  // Setup OpenGL settings.
  glDisable(GL_DEPTH_TEST);

  // Setup the modelview transformations and the viewport.
  glViewport(0, 0, w, h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

  auto host_image =
      sara::Image<sara::Rgba8>{video_stream.width(), video_stream.height()};


  // Display stuff.
  while (!glfwWindowShouldClose(window))
  {
    // Read the decoded frame and store it in a CUDA device buffer.
    sara::tic();
    video_stream.read(device_bgra_buffer);
    sara::toc("Read frame");

    // Copy the device buffer data to the pixel buffer object.
    sara::tic();
    cuda_async_copy(device_bgra_buffer, pbo);
    sara::toc("Copy to PBO");

    // Upload the pixel buffer object data to the texture object.
    sara::tic();
    pbo.bind();
    texture.bind();
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, texture.width,
                    texture.height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    // Unbind PBO.
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    sara::toc("Binding texture");

    sara::tic();
    shader.enable();
    texture.display();
    shader.disable();
    sara::toc("Display frame");

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  pbo.release();
  texture.release();
  shader.release();

  return 0;
}

int test_with_sara_graphics(int argc, char** argv)
{
  // Initialize CUDA driver.
  DriverApi::init();

  // Create a CUDA context so that we can use the GPU device.
  const auto gpu_id = 0;
  auto cuda_context = DriverApi::CudaContext{gpu_id};
  cuda_context.make_current();

  const auto video_filepath = argc < 2 ?
#ifdef _WIN32
                                       "C:/Users/David/Desktop/GOPR0542.MP4"
#elif __APPLE__
                                       "/Users/david/Desktop/Datasets/"
                                       "humanising-autonomy/turn_bikes.mp4"
#else
                                       "/home/david/Desktop/Datasets/sfm/"
                                       "Family.mp4"
#endif
                                       : argv[1];

  // Initialize a CUDA-powered video streamer object.
  auto video_stream = shakti::VideoStream{video_filepath, cuda_context};

  // Create a video frame buffer.
  DriverApi::DeviceBgraBuffer device_bgra_buffer{video_stream.width(),
                                                 video_stream.height()};


  // Open a window on which GLFW can operate.
  const auto w = video_stream.width();
  const auto h = video_stream.height();
  sara::create_window(w, h, "Video Stream");

  auto host_image_bgra =
      sara::Image<sara::Bgra8>{video_stream.width(), video_stream.height()};

  // Display stuff.
  for (;;)
  {
    // Read the decoded frame and store it in a CUDA device buffer.
    sara::tic();
    const auto has_frame = video_stream.read(device_bgra_buffer);
    sara::toc("Read frame");
    if (!has_frame)
      break;

    sara::tic();
    device_bgra_buffer.to_host(host_image_bgra);
    sara::toc("Copy to host");

    sara::tic();
    sara::display(host_image_bgra);
    sara::toc("Display");
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(test_with_sara_graphics);
  // app.register_user_main(test_with_glfw);
  return app.exec();
}
