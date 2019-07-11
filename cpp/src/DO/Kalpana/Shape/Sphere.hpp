#pragma once

#include <Eigen/Core>

#include <GL/gl.h>


namespace DO { namespace Kalpana {

  using namespace Eigen;

  class Sphere
  {
  public:
    Sphere(const Vector3f& c = Vector3f::Zero(), float r = 1.f)
      : _c{ c }
      , _r{ r }
    {
    }

    void init_gl();
    void call_gl();
    void delete_gl();

  private:
    GLuint _display_list;
    Vector3f _c;
    float _r;
  };

} /* namespace Kalpana */
} /* namespace DO */
