#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN_SIMPLE()
{
  Window w = openGLWindow(300, 300);
  setActiveWindow(w);

  SimpleTriangleMesh3f mesh;
  string filename = srcPath("../../datasets/pumpkin_tall_10k.obj");
  if (!MeshReader().readObjFile(mesh, filename))
  {
    cout << "Error reading mesh file:\n" << filename << endl;
    closeWindow();
    return EXIT_FAILURE;
  }
  cout << "Read " << filename << " successfully" << endl;

  displayMesh(mesh);

  bool quit = false;
  while (!quit)
  {
    int c = getKey();
    quit = (c==KEY_ESCAPE || c==' ');
  }
  closeWindow();

  return EXIT_SUCCESS;
}