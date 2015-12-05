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

namespace isaac
{


template<int N,int I>
struct isaac_for_each_unrolled_4_params
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename I0, typename F, typename P1, typename P2, typename P3, typename P4>
	ISAAC_HOST_DEVICE_INLINE static void call(I0 const& i0, F const& f, P1& p1, P2& p2, P3& p3, P4& p4)
	{
		f(N - I,*i0,p1,p2,p3,p4);
		isaac_for_each_unrolled_4_params<N,I-1>::call(boost::fusion::next(i0), f, p1, p2, p3, p4);
	}
};

template<int N>
struct isaac_for_each_unrolled_4_params<N,0>
{
	ISAAC_NO_HOST_DEVICE_WARNING
	template<typename It, typename F, typename P1, typename P2, typename P3, typename P4>
	ISAAC_HOST_DEVICE_INLINE static void call(It const&, F const&, P1& p1, P2& p2, P3& p3, P4& p4)
	{
	}
};

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename P1,typename P2, typename P3, typename P4>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_4_params(Sequence& seq, F const& f, P1& p1, P2& p2, P3& p3, P4& p4)
{
	typedef typename boost::fusion::result_of::begin<Sequence>::type begin;
	typedef typename boost::fusion::result_of::end<Sequence>::type end;
	isaac_for_each_unrolled_4_params
	<
		boost::fusion::result_of::distance<begin, end>::type::value,
		boost::fusion::result_of::distance<begin, end>::type::value
	>::call(boost::fusion::begin(seq), f, p1, p2, p3, p4);
}

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename P1,typename P2,typename P3>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_3_params(Sequence& seq, F const& f, P1& p1, P2& p2, P3& p3)
{
	int i = 0;
	isaac_for_each_4_params<Sequence,F,P1,int>(seq,f,p1,p2,p3,i);
}

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename P1,typename P2>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_2_params(Sequence& seq, F const& f, P1& p1, P2& p2)
{
	int i = 0;
	isaac_for_each_3_params<Sequence,F,P1,int>(seq,f,p1,p2,i);
}

ISAAC_NO_HOST_DEVICE_WARNING
template <typename Sequence, typename F,typename P1>
ISAAC_HOST_DEVICE_INLINE void isaac_for_each_1_params(Sequence& seq, F const& f, P1& p1)
{
	int i = 0;
	isaac_for_each_2_params<Sequence,F,P1,int>(seq,f,p1,i);
}


} //namespace isaac;
