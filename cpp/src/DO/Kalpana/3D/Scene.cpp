#include <DO/Kalpana/3D.hpp>


using namespace std;


namespace DO { namespace Kalpana {

  PointCloud * Scene::scatter(const vector<Vector3f>& points)
  {
    auto colors = vector<Vector3f>(points.size(), Vector3f::Ones());
    auto sizes = vector<float>(points.size(), 10.f);

    unique_ptr<SceneItem> point_cloud{ new PointCloud{ points, colors, sizes } };

    _objects.push_back(std::move(point_cloud));

    return dynamic_cast<PointCloud *>(_objects.back().get());
  }

} /* namespace Kalpana */
} /* namespace DO */
