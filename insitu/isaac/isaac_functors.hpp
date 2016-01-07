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

#include "isaac_types.hpp"
#include <boost/mpl/list.hpp>
#include <boost/mpl/int.hpp>

namespace isaac
{

namespace fus = boost::fusion;
namespace mpl = boost::mpl;

struct IsaacFunctorLength
{
    static const bool uses_parameter = false;
    static const std::string name;
    static const std::string description;
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
};

const std::string IsaacFunctorLength::name = "length";
const std::string IsaacFunctorLength::description = "Calculates the length of an input. Reduces the feature dimension to 1.";

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
};
const std::string IsaacFunctorMul::name = "mul";
const std::string IsaacFunctorMul::description = "Multiplies the input with a constant parameter. Keeps the feature dimension.";

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
};
const std::string IsaacFunctorAdd::name = "add";
const std::string IsaacFunctorAdd::description = "Summarizes the input with a constant parameter. Keeps the feature dimension.";

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
};
const std::string IsaacFunctorPow::name = "pow";
const std::string IsaacFunctorPow::description = "Calculates the power of the input with a constant exponent. Keeps the feature dimension.";

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
};
const std::string IsaacFunctorIdem::name = "idem";
const std::string IsaacFunctorIdem::description = "Does nothing. Keeps the feature dimension.";

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
};

const std::string IsaacFunctorSum::name = "sum";
const std::string IsaacFunctorSum::description = "Calculates the sum of all components. Reduces the feature dimension to 1.";

typedef fus::list <
    IsaacFunctorIdem,
    IsaacFunctorAdd,
    IsaacFunctorMul,
    IsaacFunctorLength,
    IsaacFunctorPow,
    IsaacFunctorSum
> IsaacFunctorPool;

} //namespace isaac;
