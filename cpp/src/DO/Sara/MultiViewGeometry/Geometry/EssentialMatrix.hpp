#pragma once

#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>


namespace DO { namespace Sara {

  class EssentialMatrix : public FundamentalMatrix
  {
    using base_type = FundamentalMatrix;

  public:
    using matrix_type = typename base_type::matrix_type;
    using point_type = typename base_type::point_type;
    using vector_type = point_type;

    using motion_type = std::pair<matrix_type, vector_type>;

    EssentialMatrix() = default;

    EssentialMatrix(const matrix_type& E)
      : base_type{E}
    {
    }

    auto extract_candidate_camera_motions() const
        -> std::array<motion_type, 4>;
  };

} /* namespace Sara */
} /* namespace DO */
