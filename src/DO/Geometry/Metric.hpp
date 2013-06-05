/*
 * =============================================================================
 *
 *       Filename:  Metric.hpp
 *
 *    Description:  Anisotropic distance and open ball data structures.
 *
 *        Version:  1.0
 *        Created:  04/02/2011 22:09
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_GEOMETRY_METRIC_HPP
#define DO_GEOMETRY_METRIC_HPP

namespace DO {

	template <typename T, int N>
	class SquaredRefDistance
	{
	public:
		typedef T Scalar;
		typedef Matrix<T, N, 1> Vector, Point;
    typedef Matrix<T, N, N> Matrix;

	public:
		inline SquaredRefDistance(const Matrix& m) : m_(m) {}
		inline const Matrix& mappedMatrix() const { return m_; }
		inline int dim() const { return N; }
		inline T operator()(const Vector& a, const Vector& b) const
  	{ return (b-a).dot(m_*(b-a)); }
		inline bool isQuasiIsotropic(T threshold = 0.9) const
		{
			Eigen::JacobiSVD<Matrix> svd(m_);
      return (svd.singularValues()(N-1)/svd.singularValues(0)) < threshold;
    }

	private:
		const Matrix& m_;
	};

	template <typename T, int N>
	class SquaredDistance
	{
	public:
		typedef T Scalar;
		typedef Matrix<T, N, 1> Vector, Point;
    typedef Matrix<T, N, N> Matrix;

	public:
		inline SquaredDistance(const Matrix& m) : m_(m) {}
		inline Matrix& mappedMatrix() { return m_; }
		inline const Matrix& mappedMatrix() const { return m_; }
		inline int dim() const { return N; }
		inline T operator()(const Vector& a, const Vector& b) const
		{ return (b-a).dot(m_*(b-a)); }
		inline bool isQuasiIsotropic(T threshold = 0.9) const
		{
      Eigen::JacobiSVD<Matrix> svd(m_);
      return (svd.singularValues()(N-1)/svd.singularValues(0)) < threshold;
		}

	private:
		const Matrix m_;
	};

	template <typename SquaredMetric>
	class OpenBall
	{
	public:
		typedef SquaredMetric SquaredDistance;
		typedef typename SquaredDistance::Scalar T;
		typedef typename SquaredDistance::Matrix Matrix;
		typedef typename SquaredDistance::Vector Vector, Point;

		inline OpenBall(const Point& center, T radius,
						        const SquaredDistance& squaredDistance)
		  : center_(center), radius_(radius), squaredDistance_(squaredDistance) {}

		inline const Point& center() const { return center_; }
		inline T radius() const { return radius_; }
		inline const SquaredDistance& squaredDistance() const { return squaredDistance_; }
		inline bool isInside(const Point& x) const
		{ return squaredDistance(center_, x) < radius_*radius_; }

	private:
		const Point& center_;
		const T radius_;
		const SquaredDistance& squaredDistance_;
	};

} /* namespace DO */

#endif /* DO_GEOMETRY_METRIC_HPP */