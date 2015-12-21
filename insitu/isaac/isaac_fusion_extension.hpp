/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Lesser Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <boost/fusion/include/list.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/int.hpp>

namespace isaac
{


template<int N,int I>
struct isaac_for_each_unrolled_params
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename I0, typename F, typename... P>
	ISAAC_HOST_DEVICE_INLINE static void call(I0 const& i0, F const& f, P&... p)
	{
		f(N - I,*i0,p...);
		isaac_for_each_unrolled_params<N,I-1>::call(boost::fusion::next(i0), f, p...);
	}
};

template<int N>
struct isaac_for_each_unrolled_params<N,0>
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename It, typename F, typename... P>
	ISAAC_HOST_DEVICE_INLINE static void call(It const&, F const&, P&... p)
	{
	}
};

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename... P>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_params(Sequence& seq, F const& f, P&... p)
{
	typedef typename boost::fusion::result_of::begin<Sequence>::type begin;
	typedef typename boost::fusion::result_of::end<Sequence>::type end;
	isaac_for_each_unrolled_params
	<
		boost::fusion::result_of::distance<begin, end>::type::value,
		boost::fusion::result_of::distance<begin, end>::type::value
	>::call(boost::fusion::begin(seq), f, p...);
}

template<int N,int I>
struct isaac_for_each_with_mpl_unrolled_params
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename I0, typename F, typename... P>
	ISAAC_HOST_DEVICE_INLINE static void call( I0 const& i0, F const& f, P&... p)
	{
		boost::mpl::int_<N - I> nr;
		f(nr,*i0,p...);
		isaac_for_each_with_mpl_unrolled_params<N,I-1>::call(boost::fusion::next(i0), f, p...);
	}
};

template<int N>
struct isaac_for_each_with_mpl_unrolled_params<N,0>
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename It, typename F, typename... P>
	ISAAC_HOST_DEVICE_INLINE static void call(It const&, F const&, P&... p)
	{
	}
};

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename... P>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_with_mpl_params(Sequence& seq, F const& f, P&... p)
{
	typedef typename boost::fusion::result_of::begin<Sequence>::type begin;
	typedef typename boost::fusion::result_of::end<Sequence>::type end;
	isaac_for_each_with_mpl_unrolled_params
	<
		boost::fusion::result_of::distance<begin, end>::type::value,
		boost::fusion::result_of::distance<begin, end>::type::value
	>::call( boost::fusion::begin(seq), f, p...);
}

} //namespace isaac;
