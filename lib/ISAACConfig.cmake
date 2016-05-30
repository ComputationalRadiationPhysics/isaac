# - Config file for the isaac package
# It defines the following variables
#  ISAAC_INCLUDE_DIRS - include directories for FooBar
#  ISAAC_LIBRARIES    - libraries to link against
#  ISAAC_DEFINITIONS  - necessary definitions
#  ISAAC_FOUND        - whether ISAAC was found and is useable
#
# It defines the following options
#  ISAAC_THREADING
#  ISAAC_SHOWBORDER
#  ISAAC_CUDA
#  ISAAC_ALPAKA
#  ISAAC_JPEG

###############################################################################
# ISAAC
###############################################################################
cmake_minimum_required (VERSION 3.1.0)
cmake_policy(SET CMP0048 OLD)

set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} "${ISAAC_DIR}")
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} "${ISAAC_DIR}/isaac")

set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-std=c++11")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-march=native")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-mtune=native")


###############################################################################
# MODULES
###############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ISAAC_DIR}/Modules")


###############################################################################
# OPTIONS
###############################################################################
option(ISAAC_THREADING "Do the video and metadata transmission transport in background." ON)
if (ISAAC_THREADING)
  add_definitions(-DISAAC_THREADING)
endif ()

option(ISAAC_SHOWBORDER "Debug and presentation mode, in which the sub volume borders are shown in black" OFF)
if (ISAAC_SHOWBORDER)
  add_definitions(-DISAAC_SHOWBORDER)
endif ()

option(ISAAC_CUDA "Using CUDA" ON)
option(ISAAC_ALPAKA "Using ALPKA" OFF)

if ( (NOT ISAAC_CUDA) AND (NOT ISAAC_ALPAKA) )
    message( FATAL_ERROR "At least Alpaka or Cuda have to be activated!" )
endif()

option(ISAAC_JPEG "Use JPEG compression between visualization and isaac server. Deactivating will not work with big images. And with big I am talking about bigger than 800x600." ON)
if (ISAAC_JPEG)
        find_package(JPEG)
        if (JPEG_FOUND)
            set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR})
            set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${JPEG_LIBRARY})
            set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_JPEG")
        else()
            message( WARNING "Using ISAAC without libjpeg is not recommended." )
        endif()
endif (ISAAC_JPEG)

set(ISAAC_VECTOR_ELEM "2" CACHE STRING "The amounts of elements used for vectorization. On GPU 1..2 should be fine, on CPU 4..16" )
add_definitions(-DISAAC_VECTOR_ELEM=${ISAAC_VECTOR_ELEM})

###############################################################################
# JANSSON LIB
###############################################################################
# set(JANSSON_DIR JANSSON_DIR_NOT-FOUND CACHE PATH "The location of the jansson library")
find_package (Jansson CONFIG)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${JANSSON_LIBRARIES})
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${JANSSON_INCLUDE_DIRS})


###############################################################################
# PTHREADS
###############################################################################
find_package (Threads MODULE)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# MPI LIB
################################################################################
find_package(MPI MODULE)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${MPI_C_LIBRARIES})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${MPI_CXX_LIBRARIES})


################################################################################
# IceT LIB
################################################################################
find_package (IceT CONFIG)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${ICET_CORE_LIBS})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${ICET_MPI_LIBS})
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${ICET_INCLUDE_DIRS})


################################################################################
# BOOST LIB
################################################################################
find_package(Boost 1.56.0 MODULE)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)

set(ISAAC_PRIVATE_FOUND true)

################################################################################
# CUDA LIB
################################################################################
if (ISAAC_CUDA)
    find_package( CUDA 7.0 )
    if (!CUDA_FOUND)
        set(ISAAC_PRIVATE_FOUND false)
    else()
        set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    endif()
endif()


################################################################################
# Alpaka LIB
################################################################################
if (ISAAC_ALPAKA)
    find_package(alpaka)
    if (!alpaka_FOUND)
        set(ISAAC_PRIVATE_FOUND false)
    else()
        set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
        set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${alpaka_LIBRARIES})
        set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} ${alpaka_DEFINITIONS})
        set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
        set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_ALPAKA")
        set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )
    endif()
endif()

list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(ISAAC
                                  REQUIRED_VARS ISAAC_LIBRARIES ISAAC_INCLUDE_DIRS JANSSON_LIBRARIES JANSSON_INCLUDE_DIRS CMAKE_THREAD_LIBS_INIT MPI_CXX_FOUND ICET_CORE_LIBS ICET_MPI_LIBS ICET_INCLUDE_DIRS Boost_FOUND ISAAC_PRIVATE_FOUND)
