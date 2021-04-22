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

#define ISAAC_START_TIME_MEASUREMENT(unique_name, time_function)                                                      \
    uint64_t BOOST_PP_CAT(__tm_start_, unique_name) = time_function;
#define ISAAC_STOP_TIME_MEASUREMENT(result, operand, unique_name, time_function)                                      \
    result operand time_function - BOOST_PP_CAT(__tm_start_, unique_name);

#ifdef ISAAC_THREADING
#    define ISAAC_WAIT_VISUALIZATION                                                                                  \
        if(visualizationThread)                                                                                       \
        {                                                                                                             \
            pthread_join(visualizationThread, NULL);                                                                  \
            visualizationThread = 0;                                                                                  \
        }
#else
#    define ISAAC_WAIT_VISUALIZATION                                                                                  \
        {                                                                                                             \
        }
#endif

#define ISAAC_HANDLE_EPIPE(add, n, sockfd, readThread)                                                                \
    if(add < 0)                                                                                                       \
    {                                                                                                                 \
        pthread_join(readThread, NULL);                                                                               \
        readThread = 0;                                                                                               \
        sockfd = 0;                                                                                                   \
        return n;                                                                                                     \
    }
