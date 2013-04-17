// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_IMAGEPROCESSING_DIFFERENTIAL_HPP
#define DO_IMAGEPROCESSING_DIFFERENTIAL_HPP

namespace DO {

#define LocatorType(T, N, StorageOrder, IsConstant) \
  typename Locator<typename Meta::Choose<IsConstant, const T, T>::Type, N, StorageOrder>

	template <int N, int Axis>
	struct Differential
	{
		DO_STATIC_ASSERT(Axis >= 0 && Axis < N,
      AXIS_MUST_NONNEGATIVE_AND_LESS_THAN_N);

		template <typename T, int StorageOrder>
		static inline T eval_derivative(Locator<T, N, StorageOrder>& loc)
		{
			return (loc.template axis<Axis>()[1] - loc.template axis<Axis>()[-1])
				/ static_cast<T>(2);
		}

		template <typename T, int StorageOrder>
		static inline T accumulate_neighbor_values(Locator<T, N, StorageOrder>& loc)
		{
			return loc.template axis<Axis>()[1]+loc.template axis<Axis>()[-1]
				+ Differential<N, Axis-1>::accumulate_neighbor_values(loc);
		}

		template <typename T, int StorageOrder>
		static inline void eval_gradient(Matrix<T, N, 1>& g,
                                     Locator<const T, N, StorageOrder>& loc)
		{
			g[Axis] = eval_derivative(loc);
			Differential<N, Axis-1>::eval_gradient(g, loc);
		}

		template <typename T, int StorageOrder>
		static inline T eval_laplacian(Locator<T, N, StorageOrder>& loc)
		{ return accumulate_neighbor_values(loc) - 2*N*(*loc); }

		template <typename T, int StorageOrder>
		static inline T eval_divergence(Locator<Matrix<T, N, 1>, N, StorageOrder>& loc)
		{
			return eval_derivative(loc)
				+ Differential<N, Axis-1>::eval_divergence(loc);
		}
	};

	template <int N>
	struct Differential<N,0>
	{
		template <typename T, int StorageOrder>
		static inline T eval_derivative(Locator<T, N, StorageOrder>& loc)
		{
			return (loc.template axis<0>()[1] - loc.template axis<0>()[-1])
				/ static_cast<T>(2);
		}

		template <typename T, int StorageOrder>
		static inline T accumulate_neighbor_values(Locator<T, N, StorageOrder>& loc)
		{ return loc.template axis<0>()[1] + loc.template axis<0>()[-1]; }

		template <typename T, int StorageOrder>
		static inline void eval_gradient(Matrix<T, N, 1>& g, Locator<const T, N, StorageOrder>& loc)
		{ g[0] = eval_derivative(loc); }

		template <typename T, int StorageOrder>
		static inline T eval_divergence(Locator<Matrix<T, N, 1>, N, StorageOrder>& loc)
		{ return eval_derivative(loc); }
	};

	template <typename T, int N = 2>
	struct ComputeGradient
	{
		typedef T Scalar;
		typedef Matrix<T, N, 1> Vector;
		typedef Matrix<int, N, 1> Coords;
		typedef Image<T, N> ScalarField;
		typedef Image<Vector, N> VectorField;
		typedef typename ScalarField::ConstLocator ScalarLocator;
		typedef typename VectorField::Locator VectorLocator;

		inline ComputeGradient(const ScalarField& scalarField)
      : scalar_field_(scalarField) {}

		inline void operator()(Vector& g, ScalarLocator& loc) const
		{ Differential<N, N-1>::eval_gradient(g, loc); }

		inline void operator()(Vector& g, const Coords& p) const
		{ operator()(g, scalar_field_.begin_locator(p)); }

		void operator()(VectorField& dst) const
		{
			if (dst.sizes() != scalar_field_.sizes())
				dst.resize(scalar_field_.sizes());

			ScalarLocator src_loc(scalar_field_.begin_locator());
			VectorLocator dst_loc(dst.begin_locator());
			for ( ; src_loc != src_loc.end(); ++src_loc, ++dst_loc)
				operator()(*dst_loc, src_loc);
		};

		VectorField operator()() const
		{
			VectorField gradField;
			operator()(gradField);
			return gradField;
		}

		const ScalarField& scalar_field_;
	};

	template <typename T, int N = 2>
	struct ComputeLaplacian
	{
		typedef Matrix<T, N, 1> Vector, Coords;
		typedef Image<T, N> ScalarField;
		typedef typename ScalarField::Locator Locator;
    typedef typename ScalarField::ConstLocator ConstLocator;

		inline ComputeLaplacian(const ScalarField& scalarField)
			: scalar_field_(scalarField) {}

		inline T operator()(ConstLocator& loc) const
		{ return Differential<N, N-1>::eval_laplacian(loc); }

		inline T operator()(const Coords& p) const
		{ return Differential<N, 0>::eval_laplacian(scalar_field_.locator(p)); }

		void operator()(ScalarField& dst) const
		{
			ConstLocator src_loc(scalar_field_.begin_locator());
			Locator dst_loc(dst.begin_locator());

			for ( ; src_loc != src_loc.end(); ++src_loc, ++dst_loc)
				*dst_loc = operator()(src_loc);
		};

		ScalarField operator()() const
		{
			ScalarField lapField;
			operator()(lapField);
			return lapField;
		}

		const ScalarField& scalar_field_;
	};

  // Redo that: I don't care about N > 3...
  template <typename T, int N = 2>
  struct ComputeHessian
  {
    typedef Matrix<int, N, 1> Coords, Vector;
    typedef Image<T, N> ScalarField;
    typedef Matrix<T, N, N> Matrix;
    typedef Image<Matrix, N> MatrixField;
    typedef typename ScalarField::ConstLocator ScalarLocator;
    typedef typename MatrixField::Locator MatrixLocator;

		inline ComputeHessian(const ScalarField& scalarField)
			: scalar_field_(scalarField) {}

		void operator()(Matrix& H, const ScalarLocator& loc) const
    {
      /* Just a reminder:
       *
       * ixx = ( i(x+1,y) - 2*i(x,y) + i(x-1,y) ) / static_cast<T>(4);
       * iyy = ( i(x,y+1) - 2*i(x,y) + i(x,y-1) ) / static_cast<T>(4);
       * ixy = ( (i(x+1,y+1) - i(x-1,y+1)) - (i(x+1,y-1) - i(x-1,y-1)) ) 
       *   / static_cast<T>(4);
       */
      for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j)
        {
          if (i == j)
            H(i,i) = ( loc(delta(i,1)) - 2*(*loc) + loc(delta(i,-1)) )
                   / static_cast<T>(4);
          else
          {
            H(i,j) = (  ( loc(delta(i,1,j, 1)) - loc(delta(i,-1,j, 1)) )
                      - ( loc(delta(i,1,j,-1)) - loc(delta(i,-1,j,-1)) ) )
                         / static_cast<T>(4);
            H(j,i) = H(i,j);
          }
        }
    }

		Matrix operator()(ScalarLocator& loc) const
    {
      Matrix H;
      operator()(H, loc);
      return H;
    }

		inline Matrix operator()(const Coords& p) const
    { return operator()(scalar_field_.locator(p)); }

		void operator()(MatrixField& dst) const
		{
      if (dst.sizes() != scalar_field_.sizes())
        dst.resize(scalar_field_.sizes());

			ScalarLocator src_loc(scalar_field_.begin_locator());
			MatrixLocator dst_loc(dst.begin_locator());
			for ( ; src_loc != src_loc.end(); ++src_loc, ++dst_loc)
				*dst_loc = operator()(src_loc);
		};

		MatrixField operator()() const
		{
			MatrixField hessianField;
			operator()(hessianField);
			return hessianField;
		}

    inline Coords delta(int i, int dx, int j, int dy) const
    {
      Coords unit(Coords::Zero());
      unit[i] = dx; unit[j] = dy;
      return unit;
    }

    inline Coords delta(int i, int dx) const
    { return Coords::Unit(i)*dx; }

		const ScalarField& scalar_field_;
  };

	//! helper functions
	template <typename T, int N>
	void grad(Image<Matrix<T,N,1>, N>& dst, const Image<T, N>& src)
	{
		ComputeGradient<T,N> computeGradient(src);
		computeGradient(dst);
	}

	template <typename T, int N>
	void del2(Image<T>& dst, const Image<T, N>& src)
	{
		ComputeLaplacian<T, N> computeLaplacian(src);
		computeLaplacian(dst);
	}

	template <typename T, int N>
	void squaredNorm(Image<T, N>& dst, const Image<Matrix<T,N,1>, N>& src)
	{
    if (dst.sizes() != src.sizes())
      dst.resize(src.sizes());

		typedef typename Image<T, N>::Locator ScalarLoc;
		typedef typename Image<Matrix<T,N,1>, N>::ConstLocator VectorLoc;

		ScalarLoc dst_loc(dst.begin_locator());
		VectorLoc src_loc(src.begin_locator());
		for ( ; dst_loc != dst_loc.end(); ++dst_loc, ++src_loc)
			*dst_loc = src_loc->squaredNorm();
	}

	template <typename T, int N>
	void blueNorm(Image<T, N>& dst, const Image<Matrix<T,N,1>, N>& src)
	{
		typedef typename Image<T, N>::Locator ScalarLoc;
		typedef typename Image<Matrix<T,N,1>, N>::ConstLocator VectorLoc;

		ScalarLoc dst_loc(dst.begin_locator());
		VectorLoc src_loc(src.begin_locator());
		for ( ; dst_loc != dst_loc.end(); ++dst_loc, ++src_loc)
			*dst_loc = src_loc->blueNorm();
	}

	template <typename T, int N>
	void stableNorm(Image<T, N>& dst, const Image<Matrix<T,N,1>, N>& src)
	{
		typedef typename Image<T, N>::Locator ScalarLoc;
		typedef typename Image<Matrix<T,N,1>, N>::ConstLocator VectorLoc;

		ScalarLoc dst_loc(dst.begin_locator());
		VectorLoc src_loc(src.begin_locator());
		for ( ; dst_loc != dst_loc.end(); ++dst_loc, ++src_loc)
			*dst_loc = src_loc->stableNorm();
	}

	template <typename T>
	void orientation(Image<T>& dst, const Image<Matrix<T, 2, 1> >& src)
	{
		typedef typename Image<T>::Locator ScalarLoc;
		typedef typename Image<Matrix<T,2,1> >::ConstLocator VectorLoc;

		ScalarLoc dst_loc(dst.begin_locator());
		VectorLoc src_loc(src.begin_locator());
		for ( ; dst_loc != dst_loc.end(); ++dst_loc, ++src_loc)
			*dst_loc = std::atan2(src_loc->y(), src_loc->x());
	}

  template <typename T, int N>
  void hessian(Image<Matrix<T, N, N> >& dst, const Image<T, N>& src)
  {
    ComputeHessian<T, N> computeHessian(src);
    computeHessian(dst);
  }

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DIFFERENTIAL_HPP */
