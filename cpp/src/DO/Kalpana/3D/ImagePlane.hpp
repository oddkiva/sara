#pragma once


class ImageGeometry
{
public:
  void initialize();

  // Can be shared.
  QOpenGLShaderProgram *m_program{nullptr};
  Tensor_<float, 2> m_vertices;
  Tensor_<unsigned int, 2> m_triangles;
};

class ImageTexture 
{
public:
  void initialize();

  QOpenGLTexture* m_texture{nullptr};
};


class ImagePlane
{
public:
  ImagePlane() = default;
  void initialize();
  void render();

private:
  ImageGeometry *image_geometry{nullptr};
  ImagePlane *image_texture{nullptr};
};
