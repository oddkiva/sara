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

#ifndef DO_IMAGEPROCESSING_LINEARFILTERING_HPP
#define DO_IMAGEPROCESSING_LINEARFILTERING_HPP

namespace DO {
    
	enum PaddingType { ZeroPadding, BorderReplication };

	template <typename T>
	void convolveBuffer(T *buffer,
					    const typename ColorTraits<T>::ChannelType *kernel,
						int rsize, int ksize)
	{
		T *bp;
		T *b = buffer;
		const typename ColorTraits<T>::ChannelType *kp;

		for (int i = 0; i < rsize; ++i)
		{
			bp = b;
			kp = kernel;

			T sum(ColorTraits<T>::zero());
			for (int j = 0; j < ksize; j++)
				sum += *bp++ * *kp++;
			
			*b++ = sum;
		}
	}

	// ====================================================================== //
	// Linear filters.
	template <typename T>
	void applyRowBasedFilter(Image<T>& dst, const Image<T>& src,
							 const typename ColorTraits<T>::ChannelType *kernel,
							 int kSize)
	{
		const int w = src.width();
		const int h = src.height();
		const int halfSize = kSize/2;
		Image<T> buffer(w+halfSize*2,1);

		// Resize if necessary.
		if (dst.sizes() != src.sizes())
			dst.resize(w,h);


		for (int y = 0; y < h; ++y) {
			// Copy to work array and add padding.
			for (int x = 0; x < halfSize; ++x)
				buffer(x,0) = src(0,y);
			for (int x = 0; x < w; ++x)
				buffer(halfSize+x,0) = src(x,y);
			for (int x = 0; x < halfSize; ++x)
				buffer(w+halfSize+x,0) = src(w-1,y);

			// Compute the value by convolution
			for (int x = 0; x < w; ++x) {
				dst(x,y) = ColorTraits<T>::zero();
				for (int k = 0; k < kSize; ++k)
					dst(x,y) += kernel[k]*buffer(x+k,0);
			}
		}
	}

	template <typename T>
	void applyColumnBasedFilter(Image<T>& dst, const Image<T>& src,
								const typename ColorTraits<T>::ChannelType *kernel,
								int kSize)
	{
		const int w = src.width();
		const int h = src.height();
		const int halfSize = kSize/2;
		
		// Resize if necessary.
		if (dst.sizes() != src.sizes())
			dst.resize(w,h);

		Image<T> buffer(h+halfSize*2,1);

		for (int x = 0; x < w; ++x)
		{
			// Copy to work array and add padding.
			for (int y = 0; y < halfSize; ++y)
				buffer(y,0) = src(x,0);
			for (int y = 0; y < h; ++y)
				buffer(y+halfSize,0) = src(x,y);
			for (int y = 0; y < halfSize; ++y)
				buffer(h+halfSize+y,0) = src(x,h-1);

			// Compute the value by convolution
			for (int y = 0; y < h; ++y)
			{
				dst(x,y) = ColorTraits<T>::zero();
				for (int k = 0; k < kSize; ++k)
					dst(x,y) += kernel[k]*buffer(y+k,0);
			}
		}
	}

	template <typename T>
	void applyFastRowBasedFilter(Image<T>& dst, const Image<T>& src,
								 const typename ColorTraits<T>::ChannelType *kernel,
								 int kSize)
	{
		const int w = src.width();
		const int h = src.height();
		const int halfSize = kSize/2;
		T *buffer = new T[w+halfSize*2];
		for (int y = 0; y < h; ++y)
		{
			// Copy to work array and add padding.
			for (int x = 0; x < halfSize; ++x)
				buffer[x] = src(0,y);
			for (int x = 0; x < w; ++x)
				buffer[halfSize+x] = src(x,y);
			for (int x = 0; x < halfSize; ++x)
				buffer[w+halfSize+x] = src(w-1,y);

			convolveBuffer(buffer, kernel, w, kSize);
			for (int x = 0; x < w; ++x)
				dst(x,y) = buffer[x];
		}

		delete[] buffer;
	}
	template <typename T>
	void applyFastColumnBasedFilter(Image<T>& dst, const Image<T>& src,
									const typename ColorTraits<T>::ChannelType *kernel,
									int kSize)
	{
		const int w = src.width();
		const int h = src.height();
		const int halfSize = kSize/2;

		// Resize if necessary.
		if (dst.sizes() != src.sizes())
			dst.resize(w,h);

		T *buffer = new T[h+halfSize*2];

		for (int x = 0; x < w; ++x)
		{
			for (int y = 0; y < halfSize; ++y)
				buffer[y] = src(x,0);
			for (int y = 0; y < h; ++y)
				buffer[halfSize+y] = src(x,y);
			for (int y = 0; y < halfSize; ++y)
				buffer[h+halfSize+y] = src(x,h-1);
			
			convolveBuffer(buffer, kernel, h, kSize);
			for (int y = 0; y < h; ++y)
				dst(x,y) = buffer[y];
		}

		delete[] buffer;
	}

	template <typename T>
	void applyRowDerivative(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S diff[] = { S(1), S(0), S(-1) };
		applyFastRowBasedFilter(dst, src, diff, 3);
	}

	template <typename T>
	void applyColumnDerivative(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S diff[] = { S(1), S(0), S(-1) };
		applyFastColumnBasedFilter(dst, src, diff, 3);
	}

	template <typename T>
	void applyGaussianFilter(Image<T>& dst, const Image<T>& src,
                           typename ColorTraits<T>::ChannelType sigma,
                           typename ColorTraits<T>::ChannelType gaussTruncate = 
                           typename ColorTraits<T>::ChannelType(4))
	{
		DO_STATIC_ASSERT(
      !std::numeric_limits<typename ColorTraits<T>::ChannelType >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );

		typedef typename ColorTraits<T>::ChannelType S;

		// Compute the size of the Gaussian kernel.
		int kSize = int(S(2) * gaussTruncate * sigma + S(1));
		// Make sure the Gaussian kernel is at least of size 3 and is of odd size.
		kSize = std::max(3, kSize);
		if (kSize % 2 == 0)
			++kSize;

		// Create the 1D Gaussian kernel.
		S *kernel = new S[kSize];
		S sum(0);

		// Compute the value of the Gaussian and the normalizing factor.
		for (int i = 0; i < kSize; ++i)
		{
			S x = S(i - kSize/2);
			kernel[i] = exp(-x*x/(S(2)*sigma*sigma));
			sum += kernel[i];
		}

		// Normalize the kernel.
		for (int i = 0; i < kSize; ++i)
			kernel[i] /= sum;

		applyFastRowBasedFilter(dst, src, &kernel[0], kSize);
		applyFastColumnBasedFilter(dst, dst, &kernel[0], kSize);
        
    delete[] kernel;
	}

	template <typename T>
	void applySobelFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S meanKernel[] = { S(1), S(2), S(1) };
		S diffKernel[] = { S(1), S(0), S(-1) };

		Image<T> tmp(src.sizes());
		applyFastRowBasedFilter(tmp, src, meanKernel, 3);
		applyFastColumnBasedFilter(tmp, tmp, diffKernel, 3);
		applyFastRowBasedFilter(dst, src, diffKernel, 3);
		applyFastColumnBasedFilter(dst, dst, meanKernel, 3);

		dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
	}

	template <typename T>
	void applyScharrFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S meanKernel[] = { S(3), S(10), S(3) };
		S diffKernel[] = { S(1), S(0), S(-1) };

		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());
		Image<T> tmp(src.sizes());
		applyFastRowBasedFilter(tmp, src, meanKernel, 3);
		applyFastColumnBasedFilter(tmp, tmp, diffKernel, 3);
		applyFastRowBasedFilter(dst, src, diffKernel, 3);
		applyFastColumnBasedFilter(dst, dst, meanKernel, 3);

		dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
	}

	template <typename T>
	void applyPrewittFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S meanKernel[] = { S( 1), S(1), S(1) };
		S diffKernel[] = { S(-1), S(0), S(1) };
		
		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());
		Image<T> tmp(src.sizes());
		applyFastRowBasedFilter(tmp, src, meanKernel, 3);
		applyFastColumnBasedFilter(tmp, tmp, diffKernel, 3);
		applyFastRowBasedFilter(dst, src, diffKernel, 3);
		applyFastColumnBasedFilter(dst, dst, meanKernel, 3);
		
		dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
	}

	// ====================================================================== //
	// Non-separable filter functions.
	template <typename T>
	void apply2DNonSeparableFilter(Image<T>& dst, const Image<T>& src,
                                 const typename ColorTraits<T>::ChannelType *kernel,
                                 int kWidth, int kHeight)
	{
		typedef typename Image<T>::Coords Coords;
		
		const int hkWidth = kWidth/2;
		const int hkHeight = kHeight/2;
		const int w = src.width();
		const int h = src.height();

		Coords sizes(src.sizes()+Coords(hkWidth*2, hkHeight*2));
		Image<T> work(sizes);
		const int workw = work.width();
		const int workh = work.height();
		
		for (int y = 0; y < workh; ++y) {
			for (int x = 0; x < workw; ++x) {
				// North-West
				if (x < hkWidth && y < hkHeight)
					work(x,y) = src(0,0);
				// West
				if (x < hkWidth && hkHeight <= y && y < h+hkHeight)
					work(x,y) = src(0,y-hkHeight);
				// South-West
				if (x < hkWidth && y >= h+hkHeight)
					work(x,y) = src(0,h-1);
				// North
				if (hkWidth <= x && x < w+hkWidth && y < hkHeight)
					work(x,y) = src(x-hkWidth,0);
				// South
				if (hkWidth <= x && x < w+hkWidth && y >= h+hkHeight)
					work(x,y) = src(x-hkWidth,h-1);
				// North-East
				if (x >= w+hkWidth && y >= h+hkHeight)
					work(x,y) = src(w-1,0);
				// East
				if (x >= w+hkWidth && hkHeight <= y && y < h+hkHeight)
					work(x,y) = src(w-1,y-hkHeight);
				// South-East
				if (x >= w+hkWidth && y >= h+hkHeight)
					work(x,y) = src(w-1,h-1);
				// Middle
				if (hkWidth <= x && x < w+hkWidth && hkHeight <= y && y < h+hkHeight)
					work(x,y) = src(x-hkWidth,y-hkHeight);
			}
		}

		// Resize if necessary.
		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());

		// Convolve
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				T val(ColorTraits<T>::zero());
				for (int yy = 0; yy < kHeight; ++yy)
					for (int xx = 0; xx < kWidth; ++xx)
						val += work(x+xx, y+yy)
							 * kernel[yy*kWidth + xx];
				dst(x,y) = val;
			}
		}
	}

	template <typename T>
	void applyLaplacianFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S kernel[9] = {
			S(0), S( 1), S(0),
			S(1), S(-4), S(1),
			S(0), S( 1), S(0)
		};
		apply2DNonSeparableFilter(dst, src, kernel, 3, 3);
	}

	template <typename T>
	void applyRobertsCrossFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
		S k1[] = { 
			S( 1), S( 0),
			S( 0), S(-1)
		};
		S k2[] = { 
			S( 0), S( 1),
			S(-1), S( 0)
		};

		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());
		Image<T> tmp;
		apply2DNonSeparableFilter(tmp, src, k1, 2, 2);
		apply2DNonSeparableFilter(dst, src, k2, 2, 2);
		dst.array() = (dst.array().abs2()+ tmp.array().abs2()).sqrt();
	}

	template <typename T>
	void applyKirschFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
    DO_STATIC_ASSERT(
      !std::numeric_limits<typename ColorTraits<T>::ChannelType >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );
		S h1[9] = {
			S(-3)/S(15), S(-3)/S(15), S( 5)/S(15),
			S(-3)/S(15), S( 0)      , S( 5)/S(15),
			S(-3)/S(15), S(-3)/S(15), S( 5)/S(15)
		};

		S h2[9] = {
			S(-3)/S(15), S(-3)/S(15), S(-3)/S(15),
			S(-3)/S(15), S( 0)      , S(-3)/S(15),
			S( 5)/S(15), S( 5)/S(15), S( 5)/S(15)
		};

		S h3[9] = {
			S(-3)/S(15), S(-3)/S(15), S(-3)/S(15),
			S( 5)/S(15), S( 0)      , S(-3)/S(15),
			S( 5)/S(15), S( 5)/S(15), S(-3)/S(15)
		};

		S h4[9] = {
			S( 5)/S(15), S( 5)/S(15), S(-3)/S(15),
			S( 5)/S(15), S( 0)      , S(-3)/S(15),
			S(-3)/S(15), S(-3)/S(15), S(-3)/S(15)
		};

		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());
		Image<T> tmp(src.sizes());
		apply2DNonSeparableFilter(tmp, src, h1, 3, 3);
		dst.array() = tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h2, 3, 3);
		dst.array() += tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h3, 3, 3);
		dst.array() += tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h4, 3, 3);
		dst.array() += tmp.array().abs();
		//dst.array().sqrt();
	}

	template <typename T>
	void applyRobinsonFilter(Image<T>& dst, const Image<T>& src)
	{
		typedef typename ColorTraits<T>::ChannelType S;
    DO_STATIC_ASSERT(
      !std::numeric_limits<typename ColorTraits<T>::ChannelType >::is_integer,
      CHANNEL_TYPE_MUST_NOT_BE_INTEGRAL );
		S h1[9] = {
			S(-1)/S(5), S( 1)/S(5), S( 1)/S(5),
			S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
			S(-1)/S(5), S( 1)/S(5), S( 1)/S(5)
		};

		S h2[9] = {
			S(-1)/S(5), S(-1)/S(5), S(-1)/S(5),
			S( 1)/S(5), S(-2)/S(5), S( 1)/S(5),
			S( 1)/S(5), S( 1)/S(5), S( 1)/S(5)
		};

		S h3[9] = {
			S( 1)/S(5), S( 1)/S(5), S( 1)/S(5),
			S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
			S(-1)/S(5), S(-1)/S(5), S( 1)/S(5)
		};

		S h4[9] = {
			S(-1)/S(5), S(-1)/S(5), S( 1)/S(5),
			S(-1)/S(5), S(-2)/S(5), S( 1)/S(5),
			S( 1)/S(5), S( 1)/S(5), S( 1)/S(5)
		};

		if (dst.sizes() != src.sizes())
			dst.resize(src.sizes());
		Image<T> tmp(src.sizes());
		apply2DNonSeparableFilter(tmp, src, h1, 3, 3);
		dst.array() = tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h2, 3, 3);
		dst.array() += tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h3, 3, 3);
		dst.array() += tmp.array().abs();
		apply2DNonSeparableFilter(tmp, src, h4, 3, 3);
		dst.array() += tmp.array().abs();
		//dst.array().sqrt();
	}

	// ====================================================================== //
	// Helper functions for linear filtering
	template <typename T>
	inline Image<T> rowDerivative(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyRowDerivative(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> columnDerivative(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyColumnDerivative(dst, src);
		return dst;
	}

	template <typename T, typename S>
	inline Image<T> gaussian(const Image<T>& src, S sigma, S gaussTruncate = S(4))
	{
		Image<T> dst(src.sizes());
		applyGaussianFilter(dst, src, sigma, gaussTruncate);
		return dst;
	}

	template <typename T>
	inline Image<T> sobel(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applySobelFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> scharr(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyScharrFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> prewitt(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyPrewittFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> robertsCross(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyRobertsCrossFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> laplacian(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyLaplacianFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> kirsch(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyKirschFilter(dst, src);
		return dst;
	}

	template <typename T>
	inline Image<T> robinson(const Image<T>& src)
	{
		Image<T> dst(src.sizes());
		applyRobinsonFilter(dst, src);
		return dst;
	}	
}

#endif /* DO_IMAGEPROCESSING_LINEARFILTERING_HPP */