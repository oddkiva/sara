// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#ifndef DO_GROWREGION_GROWTHPARAMS_HPP
#define DO_GROWREGION_GROWTHPARAMS_HPP

namespace DO {

  //! Simple criterion to test if the triangle is too flat
  class TriangleFlatness
  {
  public:
    TriangleFlatness(double lowestAngleDegree, double secondLowestDegree)
      : lb(std::cos(toRadian(lowestAngleDegree)))
      , lb2(std::cos(toRadian(secondLowestDegree))) {}

    inline bool operator()(const Point2d& a, const Point2d& b, const Point2d& c) const
    { return !isNotFlat(a, b, c); }

    bool isNotFlat(const Point2d& a, const Point2d& b, const Point2d& c) const
    {
      Vector2d d[3] = { (b-a), (c-a), (c-b) };
      for (int i = 0; i < 3; ++i)
        d[i].normalize();
      // Find the two smallest angles. They necessarily have non negative
      // dot products.
      double dot[3] = { d[0].dot(d[1]), d[1].dot(d[2]), -d[2].dot(d[0]) };
      // Sort dot products in increasing order.
      std::sort(dot, dot+3);
      // We need to consider two cases:
      // 1. All the dot products are non negative.
      //    Then the three angles are less than 90 degrees.
      // 2. One dot product is negative. It corresponds to the greatest 
      //    angle of the triangle.
      // In the end, the smallest angles have the greatest cosines which 
      // are in both cases dot[1] and dot[2] with dot[1] < dot[2].
      // In our case dot[1] <= cos(40°)=lb2 and dot[2] <= cos(30°)=lb1
      return (lb2 >= dot[1] && lb >= dot[2]);
    }

  private:
    const double lb;
    const double lb2;
  };


  class PredParams
  {
  public:
    PredParams(int featType,
               double delta_x, double delta_S_x, double delta_theta,
               double squaredRhoMin)
      : feat_type_(featType)
      , delta_x_(delta_x), delta_S_x_(delta_S_x), delta_theta_(delta_theta)
    {
    }

    int featType() const { return feat_type_; }
    double deltaX() const { return delta_x_; }
    double deltaSx() const { return delta_S_x_; }
    double deltaTheta() const { return delta_theta_; }
    double squaredRhoMin() const { return squared_rho_min_; }

  private:
    int feat_type_;
    double delta_x_, delta_S_x_, delta_theta_;
    double squared_rho_min_;
  };

  class GrowthParams
  {
  public:
    GrowthParams(size_t K = 80, double rhoMin = 0.5, 
                 double angleDeg1 = 15, double angleDeg2 = 25)
      : K_(K), rho_min_(rhoMin)
      , flat_triangle_test_(angleDeg1, angleDeg2)
    {}
    static GrowthParams defaultGrowingParams() { return GrowthParams(); };
    void addPredParams(const PredParams& params) { pf_params_.push_back(params); }
    size_t K() const { return K_; }
    double rhoMin() const { return rho_min_; }
    const PredParams& pfParams(size_t i) const
    { 
      if (i >= pf_params_.size())
      {
        const char *msg = "Fatal Error: pf_params_[i] out of bounds!";
        throw std::out_of_range(msg);
      }
      return pf_params_[i];
    }
    bool isFlat(const Vector2d triangle[3]) const
    { return flat_triangle_test_(triangle[0], triangle[1], triangle[2]); }

  private:
    size_t K_;
    double rho_min_;
    TriangleFlatness flat_triangle_test_;
    std::vector<PredParams> pf_params_;
  };

} /* namespace DO */

#endif /* DO_GROWREGION_GROWINGPARAMS_HPP */