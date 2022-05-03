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
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#pragma once

#include <boost/fusion/include/empty.hpp>
#include <boost/fusion/include/join.hpp>
#include <boost/fusion/include/list.hpp>
#include <boost/mpl/begin_end.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>
#include <type_traits>

namespace std
{
    /* @bug ISAAC is working with boost fusion lists, e.g. boost::fusion::cons<>
     * these types are not trivially copyable but are passed as parameters to the kernel.
     * Since alpaka 0.9.0 the requirement to be trivially copyable is correct enforced,
     * with the result that these types can not be used as kernel parameter.
     * This trait specialization is a temporary and risky workaround until boost fusion is
     * removed from ISAAC.
     * @attention This workaround can have bad effects on other libraries which using boost fusion and ISAAC.
     */
    template<typename... T>
    struct is_trivially_copyable<boost::fusion::cons<T...>> : public integral_constant<bool, true>
    {
    };
} // namespace std

namespace isaac
{
    template<int T_n, int T_i>
    struct ForEachUnrolledParams
    {
        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename T_I0, typename T_F, typename... T_P>
        ISAAC_HOST_DEVICE_INLINE static void call(T_I0 const& i0, T_F const& f, T_P&... p)
        {
            f(T_n - T_i, *i0, p...);
            ForEachUnrolledParams<T_n, T_i - 1>::call(boost::fusion::next(i0), f, p...);
        }
    };

    template<int T_n>
    struct ForEachUnrolledParams<T_n, 0>
    {
        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename T_It, typename T_F, typename... T_P>
        ISAAC_HOST_DEVICE_INLINE static void call(T_It const&, T_F const&, T_P&... p)
        {
        }
    };

    ISAAC_NO_HOST_DEVICE_WARNING
    template<typename T_Sequence, typename T_F, typename... T_P>
    ISAAC_HOST_DEVICE_INLINE void forEachParams(T_Sequence& seq, T_F const& f, T_P&... p)
    {
        typedef typename boost::fusion::result_of::begin<T_Sequence>::type begin;
        typedef typename boost::fusion::result_of::end<T_Sequence>::type end;
        ForEachUnrolledParams<
            boost::fusion::result_of::distance<begin, end>::type::value,
            boost::fusion::result_of::distance<begin, end>::type::value>::call(boost::fusion::begin(seq), f, p...);
    }

    template<int T_n, int T_i>
    struct ForEachWithMplUnrolledParams
    {
        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename T_I0, typename T_F, typename... T_P>
        ISAAC_HOST_DEVICE_INLINE static void call(T_I0 const& i0, T_F const& f, T_P&... p)
        {
            boost::mpl::int_<T_n - T_i> nr;
            f(nr, *i0, p...);
            ForEachWithMplUnrolledParams<T_n, T_i - 1>::call(boost::fusion::next(i0), f, p...);
        }
    };

    template<int T_n>
    struct ForEachWithMplUnrolledParams<T_n, 0>
    {
        ISAAC_NO_HOST_DEVICE_WARNING
        template<typename T_It, typename T_F, typename... T_P>
        ISAAC_HOST_DEVICE_INLINE static void call(T_It const&, T_F const&, T_P&... p)
        {
        }
    };

    ISAAC_NO_HOST_DEVICE_WARNING
    template<typename T_Sequence, typename T_F, typename... T_P>
    ISAAC_HOST_DEVICE_INLINE void forEachWithMplParams(T_Sequence& seq, T_F const& f, T_P&... p)
    {
        typedef typename boost::fusion::result_of::begin<T_Sequence>::type begin;
        typedef typename boost::fusion::result_of::end<T_Sequence>::type end;
        ForEachWithMplUnrolledParams<
            boost::fusion::result_of::distance<begin, end>::type::value,
            boost::fusion::result_of::distance<begin, end>::type::value>::call(boost::fusion::begin(seq), f, p...);
    }

} // namespace isaac
