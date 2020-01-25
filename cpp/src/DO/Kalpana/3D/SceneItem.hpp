#pragma once

#include <DO/Kalpana/Defines.hpp>

#include <DO/Kalpana/3D.hpp>
#include <DO/Kalpana/3D/Shader.hpp>

#include <QOpenGLFunctions_4_3_Core>

#include <Eigen/Core>

#include <vector>


namespace DO::Kalpana {

  using namespace Eigen;

  class DO_KALPANA_EXPORT SceneItem : protected QOpenGLFunctions_4_3_Core
  {
  public:
    SceneItem() = default;

    virtual ~SceneItem();

    void set_vertex_shader_source(const std::string& source);

    void set_fragment_shader_source(const std::string& source);

    //! @{
    //! @brief Must be called within an OpenGL context.
    void clear();

    void initialize_shaders();

    virtual void initialize() = 0;

    virtual void draw() = 0;
    //! @}

  protected:
    //! @{
    //! Shader sources.
    std::string _vs_source;
    std::string _fs_source;
    //! @}

    //! @{
    //! @brief Vertex data in device memory.
    GLuint _vbo{0};
    GLuint _vao{0};
    //! @}

    //! @{
    //! @brief Rendering properties
    Shader _vertex_shader;
    Shader _fragment_shader;
    ShaderProgram _shader_program;
    //! @}
  };


  class PointCloud : public SceneItem
  {
    struct Vertex
    {
      Vector3f point;
      Vector3f color;
      float size;
    };

  public:
    PointCloud() = default;

    PointCloud(const std::vector<Vector3f>& points,
               const std::vector<Vector3f>& colors,
               const std::vector<float>& sz);

    //! @{
    //! @brief Must be called within an OpengL context.
    void initialize() override;

    void draw() override;
    //! @}

  private:
    //! @brief Vertex data in host memory.
    std::vector<Vertex> _vertices;
  };

}  // namespace DO::Kalpana
