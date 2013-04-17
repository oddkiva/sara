// ========================================================================== //
// This file is part of DO, a lightweight C++ vision library.
//
// Copyright (C) 2012 David OK <david.ok8@gmail.com>
//
// DO is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// DO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// DO. If not, see <http://www.gnu.org/licenses/>.
// ========================================================================== //

//! @file
//! \brief Naive hash function for arrays. Check if this needs to be removed.
//! Anyway, this is not included in the master header file "Core.hpp".

#ifndef DO_CORE_HASH_HPP
#define DO_CORE_HASH_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifdef BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
namespace boost
#else
namespace Eigen
#endif
{
	template <typename T, int M, int N, int Options, int MaxRows, int MaxCols>
	inline std::size_t
	hash_value(const Eigen::Matrix<T, M, N, Options, MaxRows, MaxCols>& A)
	{
		std::size_t seed = 0;
		boost::hash_range(seed, A.data(), A.data()+A.rows()*A.cols());
		return seed;
	}

	template <typename T, int M, int N, int Options, int MaxRows, int MaxCols>
	inline std::size_t
	hash_value(const Eigen::Array<T, M, N, Options, MaxRows, MaxCols>& A)
	{
		std::size_t seed = 0;
		boost::hash_range(seed, A.data(), A.data()+A.rows()*A.cols());
		return seed;
	}

}

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif /* DO_CORE_HASH_HPP */