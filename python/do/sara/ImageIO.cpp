#include <boost/python.hpp>

#include <DO/Sara/ImageIO.hpp>

#include "ImageIO.hpp"
#include "Numpy.hpp"


namespace bp = boost::python;
namespace sara = DO::Sara;

using namespace std;


bp::object imread(const std::string& filepath)
{
  using namespace sara;

  auto image = Image<double>{};
  if (!imread(image, filepath))
    throw runtime_error{ "Cannot read image file: " + filepath };

  auto data = image.data();
  auto ndims = 2;
  npy_intp sizes[] = { image.height(), image.width() };
  auto py_obj = PyArray_SimpleNewFromData(ndims, sizes, NPY_DOUBLE, data);

  bp::handle<> handle{ py_obj };
  bp::numeric::array arr{ handle };

  return arr.copy();
}


void expose_image_io()
{
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  import_numpy_array();

  def("imread", &imread);
}
