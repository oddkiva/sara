#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Kalpana/3D.hpp>


using namespace std;


namespace DO { namespace Kalpana {

  PointCloud * Scene::scatter(const vector<Vector3f>& points)
  {
    auto colors = vector<Vector3f>(points.size(), Vector3f::Ones());
    auto sizes = vector<float>(points.size(), 10.f);

    _objects.emplace_back(new PointCloud(points, colors, sizes));

    return dynamic_cast<PointCloud*>(_objects.back().get());
  }

} /* namespace Kalpana */
} /* namespace DO */
