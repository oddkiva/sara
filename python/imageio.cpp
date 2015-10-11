#include <boost/python.hpp>

#include <DO/Sara/ImageIO.hpp>

#include "imageio.hpp"
#include "python.hpp"


using namespace std;

namespace sara = DO::Sara;


boost::python::object imread(const std::string& filepath)
{
  using namespace sara;
  auto image = Image<double>{};
  if (!imread(image, filepath))
    throw runtime_error{ "Cannot read image file: " + filepath };

  auto data = image.data();
  auto ndims = 2;
  npy_intp sizes[] = { image.height(), image.width() };
  auto py_obj = PyArray_SimpleNewFromData(ndims, sizes, NPY_DOUBLE, data);

  boost::python::handle<> handle{ py_obj };
  boost::python::numeric::array arr{ handle };

  return arr.copy();
}


void expose_imageio()
{
  using namespace boost::python;

  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  DO::Sara::python::import_numpy_array();

  // Create "sara.imageio" module name.
  string imageio_name{ extract<string>{
    scope().attr("__name__") + ".imageio"
  }};

  // Create "sara.imageio" module.
  object imageio_module{ handle<>{
    borrowed(PyImport_AddModule(imageio_name.c_str()))
  }};

  // Set the "sara.imageio" scope.
  scope().attr("imageio") = imageio_module;
  scope parent{ imageio_module };

  def("imread", &imread);
}
