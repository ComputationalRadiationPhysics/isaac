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

#include "isaac_types.hpp"
#include <boost/mpl/list.hpp>
#include <boost/mpl/int.hpp>

#ifndef ISAAC_FUNCTOR_LENGTH_ENABLED
    #define ISAAC_FUNCTOR_LENGTH_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_MUL_ENABLED
    #define ISAAC_FUNCTOR_MUL_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_ADD_ENABLED
    #define ISAAC_FUNCTOR_ADD_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_POW_ENABLED
    #define ISAAC_FUNCTOR_POW_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_SUM_ENABLED
    #define ISAAC_FUNCTOR_SUM_ENABLED 1
#endif

namespace isaac
{

namespace fus = boost::fusion;
namespace mpl = boost::mpl;

struct IsaacFunctorIdem
{
    static const bool uses_parameter = false;
    static const std::string name;
    static const std::string description;
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<4> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        return v;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<3> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        return v;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<2> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        return v;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        return v;
    }
    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("idem");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Does nothing. Keeps the feature dimension.");
    }
};

#if ISAAC_FUNCTOR_LENGTH_ENABLED == 1
struct IsaacFunctorLength
{
    static const bool uses_parameter = false;
    static const std::string name;
    static const std::string description;

    //Against annoying double->float casting warning with gcc5
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wnarrowing"

    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            sqrt(
            v.value.x * v.value.x +
            v.value.y * v.value.y +
            v.value.z * v.value.z +
            v.value.w * v.value.w
            )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            sqrt(
            v.value.x * v.value.x +
            v.value.y * v.value.y +
            v.value.z * v.value.z
            )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            sqrt(
            v.value.x * v.value.x +
            v.value.y * v.value.y
            )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result = { fabs( v.value.x ) };
        return result;
    }

    #pragma GCC diagnostics pop

    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("length");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Calculates the length of an input. Reduces the feature dimension to 1.");
    }
};
#endif

#if ISAAC_FUNCTOR_MUL_ENABLED == 1
struct IsaacFunctorMul
{
    static const bool uses_parameter = true;
    static const std::string name;
    static const std::string description;
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<4> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        isaac_float_dim<4> result = { v.value * p };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<3> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        isaac_float_dim<3> result =
        {
            v.value.x * p.x,
            v.value.y * p.y,
            v.value.z * p.z
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<2> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        isaac_float_dim<2> result =
        {
            v.value.x * p.x,
            v.value.y * p.y
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result = { v.value.x * p.x };
        return result;
    }
    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("mul");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Multiplies the input with a constant parameter. Keeps the feature dimension.");
    }
};
#endif

#if ISAAC_FUNCTOR_ADD_ENABLED == 1
struct IsaacFunctorAdd
{
    static const bool uses_parameter = true;
    static const std::string name;
    static const std::string description;
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<4> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        isaac_float_dim<4> result = { v.value + p };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<3> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        isaac_float_dim<3> result =
        {
            v.value.x + p.x,
            v.value.y + p.y,
            v.value.z + p.z
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<2> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        isaac_float_dim<2> result =
        {
            v.value.x + p.x,
            v.value.y + p.y
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result = { v.value.x + p.x };
        return result;
    }
    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("add");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Summarizes the input with a constant parameter. Keeps the feature dimension.");
    }
};
#endif

#if ISAAC_FUNCTOR_POW_ENABLED == 1
struct IsaacFunctorPow
{
    static const bool uses_parameter = true;
    static const std::string name;
    static const std::string description;
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<4> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        isaac_float_dim<4> result =
        {
            pow( v.value.x, p.x ),
            pow( v.value.y, p.y ),
            pow( v.value.z, p.z ),
            pow( v.value.w, p.w )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<3> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        isaac_float_dim<3> result =
        {
            pow( v.value.x, p.x ),
            pow( v.value.y, p.y ),
            pow( v.value.z, p.z )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<2> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        isaac_float_dim<2> result =
        {
            pow( v.value.x, p.x ),
            pow( v.value.y, p.y )
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result = { pow( v.value.x, p.x ) };
        return result;
    }
    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("pow");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Calculates the power of the input with a constant exponent. Keeps the feature dimension.");
    }
};
#endif

#if ISAAC_FUNCTOR_SUM_ENABLED == 1
struct IsaacFunctorSum
{
    static const bool uses_parameter = false;
    static const std::string name;
    static const std::string description;
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<4> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            v.value.x +
            v.value.y +
            v.value.z +
            v.value.w
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<3> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            v.value.x +
            v.value.y +
            v.value.z
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<2> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result =
        {
            v.value.x +
            v.value.y
        };
        return result;
    }
    ISAAC_HOST_DEVICE_INLINE
    static isaac_float_dim<1> call( const isaac_float_dim<1> v, const isaac_float4& p)
    {
        isaac_float_dim<1> result = { v.value.x };
        return result;
    }
    ISAAC_HOST_INLINE
    static std::string getName()
    {
        return std::string("sum");
    }
    ISAAC_HOST_INLINE
    static std::string getDescription()
    {
        return std::string("Calculates the sum of all components. Reduces the feature dimension to 1.");
    }
};
#endif

typedef fus::list <
    IsaacFunctorIdem
#if ISAAC_FUNCTOR_ADD_ENABLED == 1
    ,IsaacFunctorAdd
#endif
#if ISAAC_FUNCTOR_MUL_ENABLED == 1
    ,IsaacFunctorMul
#endif
#if ISAAC_FUNCTOR_LENGTH_ENABLED == 1
    ,IsaacFunctorLength
#endif
#if ISAAC_FUNCTOR_POW_ENABLED == 1
    ,IsaacFunctorPow
#endif
#if ISAAC_FUNCTOR_SUM_ENABLED == 1
    ,IsaacFunctorSum
#endif
> IsaacFunctorPool;

#define ISAAC_FUNCTOR_COUNT \
    BOOST_PP_ADD( BOOST_PP_IF( ISAAC_FUNCTOR_ADD_ENABLED, 1, 0 ), \
    BOOST_PP_ADD( BOOST_PP_IF( ISAAC_FUNCTOR_MUL_ENABLED, 1, 0 ), \
    BOOST_PP_ADD( BOOST_PP_IF( ISAAC_FUNCTOR_LENGTH_ENABLED, 1, 0 ), \
    BOOST_PP_ADD( BOOST_PP_IF( ISAAC_FUNCTOR_POW_ENABLED, 1, 0 ), \
    BOOST_PP_ADD( BOOST_PP_IF( ISAAC_FUNCTOR_SUM_ENABLED, 1, 0 ), \
    1 ) ) ) ) )

} //namespace isaac;
