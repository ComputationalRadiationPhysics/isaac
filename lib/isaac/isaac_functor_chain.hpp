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

#include "isaac_macros.hpp"
#include "isaac_fusion_extension.hpp"
#include "isaac_functors.hpp"

#include <boost/mpl/at.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/back.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/fusion/include/push_back.hpp>
#include <boost/mpl/size.hpp>

namespace isaac
{
    typedef isaac_float( * FunctorChainPointer4 )(
        isaac_float_dim< 4 >,
        isaac_int
    );

    typedef isaac_float( * FunctorChainPointer3 )(
        isaac_float_dim< 3 >,
        isaac_int
    );

    typedef isaac_float( * FunctorChainPointer2 )(
        isaac_float_dim< 2 >,
        isaac_int
    );

    typedef isaac_float( * FunctorChainPointer1 )(
        isaac_float_dim< 1 >,
        isaac_int
    );

    typedef isaac_float( * FunctorChainPointerN )(
        void *,
        isaac_int
    );

    ISAAC_CONSTANT isaac_float4 FunctorParameter[ ISAAC_MAX_SOURCES*ISAAC_MAX_FUNCTORS ];

    ISAAC_CONSTANT FunctorChainPointerN FunctionChain[ ISAAC_MAX_SOURCES ];

    template<
        int T_n
    >
    struct DestArrayStruct
    {
        isaac_int nr[T_n];
    };

    ISAAC_DEVICE_INLINE isaac_float applyFunctorChain(isaac_float_dim <1>& data, const int nr)
    {
        return reinterpret_cast<FunctorChainPointer1> ( FunctionChain[nr] )(
                *( reinterpret_cast< isaac_float_dim< 1 > * > ( &data ) ),
                nr
            );
    }

    ISAAC_DEVICE_INLINE isaac_float applyFunctorChain(isaac_float_dim <2>& data, const int nr)
    {

        return reinterpret_cast<FunctorChainPointer2> ( FunctionChain[nr] )(
                *( reinterpret_cast< isaac_float_dim< 2 > * > ( &data ) ),
                nr
            );
    }

    ISAAC_DEVICE_INLINE isaac_float applyFunctorChain(isaac_float_dim <3>& data, const int nr)
    {
        return reinterpret_cast<FunctorChainPointer3> ( FunctionChain[nr] )(
                *( reinterpret_cast< isaac_float_dim< 3 > * > ( &data ) ),
                nr
            );
    }

    ISAAC_DEVICE_INLINE isaac_float applyFunctorChain(isaac_float_dim <4>& data, const int nr)
    {
        return reinterpret_cast<FunctorChainPointer4> ( FunctionChain[nr] )(
                *( reinterpret_cast< isaac_float_dim< 4 > * > ( &data ) ),
                nr
            );
    }

    template<
        typename T_FunctorVector,
        int T_featureDim,
        int T_nr
    >
    struct FillFunctorChainPointerKernelStruct
    {
        ISAAC_DEVICE static FunctorChainPointerN
        call( isaac_int const * const bytecode )
        {
#define ISAAC_SUB_CALL( Z, I, U ) \
            if (bytecode[ISAAC_MAX_FUNCTORS-T_nr] == I) \
                return FillFunctorChainPointerKernelStruct \
                < \
                    typename boost::mpl::push_back< T_FunctorVector, typename boost::mpl::at_c<IsaacFunctorPool,I>::type >::type, \
                    T_featureDim, \
                    T_nr - 1 \
                > ::call( bytecode );
            BOOST_PP_REPEAT(
                ISAAC_FUNCTOR_COUNT,
                ISAAC_SUB_CALL,
                ~
            )
#undef ISAAC_SUB_CALL
            return NULL; //Should never be reached anyway
        }
    };


    template<
        typename T_FunctorVector,
        int T_featureDim
    >
    ISAAC_DEVICE isaac_float generateFunctorChain(
        isaac_float_dim <T_featureDim> const value,
        isaac_int const srcID
    )
    {
#define  ISAAC_LEFT_DEF( Z, I, U ) boost::mpl::at_c< T_FunctorVector, ISAAC_MAX_FUNCTORS - I - 1 >::type::call(
#define ISAAC_RIGHT_DEF( Z, I, U ) , FunctorParameter[ srcID * ISAAC_MAX_FUNCTORS + I ] )
#define  ISAAC_LEFT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_LEFT_DEF, ~)
#define ISAAC_RIGHT BOOST_PP_REPEAT( ISAAC_MAX_FUNCTORS, ISAAC_RIGHT_DEF, ~)
        // expands to: funcN( ... func1( func0( data, p[0] ), p[1] ) ... p[T_n] );
        return ISAAC_LEFT
        value
        ISAAC_RIGHT.x;
#undef ISAAC_LEFT_DEF
#undef ISAAC_LEFT
#undef ISAAC_RIGHT_DEF
#undef ISAAC_RIGHT
    }


    template<
        typename T_FunctorVector,
        int T_featureDim
    >
    struct FillFunctorChainPointerKernelStruct<
        T_FunctorVector,
        T_featureDim,
        0 //<- Specialization
    >
    {
        ISAAC_DEVICE static FunctorChainPointerN
        call( isaac_int const * const bytecode )
        {
            return reinterpret_cast<FunctorChainPointerN> ( generateFunctorChain<
                T_FunctorVector,
                T_featureDim
            > );
        }
    };



    struct FillFunctorChainPointerKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            FunctorChainPointerN * const functorChain
        ) const
        {
            isaac_int bytecode[ISAAC_MAX_FUNCTORS];
            for( int i = 0; i < ISAAC_MAX_FUNCTORS; i++ )
            {
                bytecode[i] = 0;
            }
            for( int i = 0; i < ISAAC_FUNCTOR_COMPLEX;
            i++ )
            {
                functorChain[i * 4 + 0] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        1,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functorChain[i * 4 + 1] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        2,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functorChain[i * 4 + 2] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        3,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                functorChain[i * 4 + 3] =
                    FillFunctorChainPointerKernelStruct<
                        boost::mpl::vector< >,
                        4,
                        ISAAC_MAX_FUNCTORS
                    >::call( bytecode );
                for( int j = ISAAC_MAX_FUNCTORS - 1; j >= 0; j-- )
                {
                    if( bytecode[j] < ISAAC_FUNCTOR_COUNT - 1 )
                    {
                        bytecode[j]++;
                        break;
                    }
                    else
                    {
                        bytecode[j] = 0;
                    }
                }
            }
        }
    };

    template<
        int T_count,
        typename T_Dest
    >
    struct UpdateFunctorChainPointerKernel
    {
        template<
            typename T_Acc
        >
        ISAAC_DEVICE void operator()(
            T_Acc const & acc,
            FunctorChainPointerN * const functorChainChooseDevice,
            FunctorChainPointerN const * const functorChain,
            T_Dest dest
        ) const
        {
            for( int i = 0; i < T_count; i++ )
            {
                functorChainChooseDevice[i] = functorChain[dest.nr[i]];
            }
        }
    };
}