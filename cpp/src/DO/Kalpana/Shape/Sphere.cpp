#include <vector>

#include <GL/gl.h>

#include <DO/Kalpana/Shape/Sphere.hpp>


using namespace std;
using namespace Eigen;


namespace DO { namespace Kalpana {

  void Sphere::init_gl()
  {
    auto slices = 36;
    auto stacks = 18;

    vector<vector<Vector3f>> points(2+stacks);
    for (int i = 0; i < stacks; ++i)
      points.resize(slices);

    for (int i = 0; i < stacks; ++i)
    {
      for (int j = 0; j < slices; ++j)
      {
      }
    }

    _display_list = glGenLists(1);

    glNewList(_display_list);
    glEndList();
  }

  void Sphere::delete_gl()
  {
    _display_list = glDeleteLists(_display_list, 1);
  }

} /* namespace Kalpana */
} /* namespace DO */
