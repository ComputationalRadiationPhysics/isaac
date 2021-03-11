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

#include <boost/fusion/include/list.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/list.hpp>

#ifndef ISAAC_FUNCTOR_LENGTH_ENABLED
#    define ISAAC_FUNCTOR_LENGTH_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_MUL_ENABLED
#    define ISAAC_FUNCTOR_MUL_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_ADD_ENABLED
#    define ISAAC_FUNCTOR_ADD_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_POW_ENABLED
#    define ISAAC_FUNCTOR_POW_ENABLED 1
#endif
#ifndef ISAAC_FUNCTOR_SUM_ENABLED
#    define ISAAC_FUNCTOR_SUM_ENABLED 1
#endif

namespace isaac
{
    struct IsaacFunctorIdem
    {
        static const bool usesParameter = false;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<N> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            return v;
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("idem");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string("Does nothing. Keeps the feature dimension.");
        }
    };

#if ISAAC_FUNCTOR_LENGTH_ENABLED == 1
    struct IsaacFunctorLength
    {
        static const bool usesParameter = false;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<1> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            return isaac_float_dim<1>(glm::length(v));
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("length");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string("Calculates the length of an input. Reduces the feature dimension to 1.");
        }
    };
#endif

#if ISAAC_FUNCTOR_MUL_ENABLED == 1
    struct IsaacFunctorMul
    {
        static const bool usesParameter = true;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<N> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            return v * isaac_float_dim<N>(p);
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("mul");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string("Multiplies the input with a constant parameter. Keeps the feature dimension.");
        }
    };
#endif

#if ISAAC_FUNCTOR_ADD_ENABLED == 1
    struct IsaacFunctorAdd
    {
        static const bool usesParameter = true;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<N> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            return v + isaac_float_dim<N>(p);
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("add");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string("Summarizes the input with a constant parameter. Keeps the feature dimension.");
        }
    };
#endif

#if ISAAC_FUNCTOR_POW_ENABLED == 1
    struct IsaacFunctorPow
    {
        static const bool usesParameter = true;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<N> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            return glm::pow(v, isaac_float_dim<N>(p));
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("pow");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string(
                "Calculates the power of the input with a constant exponent. Keeps the feature dimension.");
        }
    };
#endif

#if ISAAC_FUNCTOR_SUM_ENABLED == 1
    struct IsaacFunctorSum
    {
        static const bool usesParameter = false;
        static const std::string name;
        static const std::string description;

        template<int N>
        ISAAC_HOST_DEVICE_INLINE static isaac_float_dim<1> call(const isaac_float_dim<N> v, const isaac_float4& p)
        {
            isaac_float_dim<1> result;

            for(int i = 0; i < N; ++i)
            {
                result += v[i];
            }
            return result;
        }
        ISAAC_HOST_INLINE static std::string getName()
        {
            return std::string("sum");
        }
        ISAAC_HOST_INLINE static std::string getDescription()
        {
            return std::string("Calculates the sum of all components. Reduces the feature dimension to 1.");
        }
    };
#endif

    typedef boost::fusion::list<
        IsaacFunctorIdem
#if ISAAC_FUNCTOR_ADD_ENABLED == 1
        ,
        IsaacFunctorAdd
#endif
#if ISAAC_FUNCTOR_MUL_ENABLED == 1
        ,
        IsaacFunctorMul
#endif
#if ISAAC_FUNCTOR_LENGTH_ENABLED == 1
        ,
        IsaacFunctorLength
#endif
#if ISAAC_FUNCTOR_POW_ENABLED == 1
        ,
        IsaacFunctorPow
#endif
#if ISAAC_FUNCTOR_SUM_ENABLED == 1
        ,
        IsaacFunctorSum
#endif
        >
        IsaacFunctorPool;

#define ISAAC_FUNCTOR_COUNT                                                                                           \
    BOOST_PP_ADD(                                                                                                     \
        BOOST_PP_IF(ISAAC_FUNCTOR_ADD_ENABLED, 1, 0),                                                                 \
        BOOST_PP_ADD(                                                                                                 \
            BOOST_PP_IF(ISAAC_FUNCTOR_MUL_ENABLED, 1, 0),                                                             \
            BOOST_PP_ADD(                                                                                             \
                BOOST_PP_IF(ISAAC_FUNCTOR_LENGTH_ENABLED, 1, 0),                                                      \
                BOOST_PP_ADD(                                                                                         \
                    BOOST_PP_IF(ISAAC_FUNCTOR_POW_ENABLED, 1, 0),                                                     \
                    BOOST_PP_ADD(BOOST_PP_IF(ISAAC_FUNCTOR_SUM_ENABLED, 1, 0), 1)))))

} // namespace isaac
