# - Config file for the isaac package
# It defines the following variables
#  ISAAC_INCLUDE_DIRS - include directories for FooBar
#  ISAAC_LIBRARIES    - libraries to link against
#  ISAAC_DEFINITIONS  - necessary definitions
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
cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0048 OLD)
project (ISAAC)

set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} "${ISAAC_DIR}/insitu")
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} "${ISAAC_DIR}/insitu/isaac")

set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-std=c++11")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-march=native")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-mtune=native")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_MAX_FUNCTORS=3")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_FUNCTOR_POW_ENABLED=0")


###############################################################################
# MODULES
###############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ISAAC_DIR}/Modules")


###############################################################################
# OPTIONS
###############################################################################
option(ISAAC_THREADING "Do the video and metadata transmission transport in background. May be buggy, but seems to work fine!" OFF)
if (ISAAC_THREADING)
  add_definitions(-DISAAC_THREADING)
endif ()

option(ISAAC_SHOWBORDER "Debug and presentation mode, in which the sub volume borders are shown in black" OFF)
if (ISAAC_SHOWBORDER)
  add_definitions(-DISAAC_SHOWBORDER)
endif ()

option(ISAAC_CUDA "Build the example using cuda." ON)

option(ISAAC_ALPAKA "Build the example using alpaka." OFF)

option(ISAAC_JPEG "Use JPEG compression between visualization and isaac server. Deactivating will not work with big images. And with big I am talking about bigger than 800x600." ON)
if (ISAAC_JPEG)
        find_package(JPEG REQUIRED)
        set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR})
        set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${JPEG_LIBRARY})
        set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_JPEG")
endif (ISAAC_JPEG)


###############################################################################
# PKGCONFIG
###############################################################################
find_package (PkgConfig MODULE REQUIRED)


###############################################################################
# JANSSON LIB
###############################################################################
find_package (Jansson MODULE REQUIRED)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${JANSSON_LIBRARIES})
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${JANSSON_INCLUDE_DIRS})


###############################################################################
# LIBWEBSOCKETS
###############################################################################
find_package(LibWebSockets MODULE REQUIRED)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${LIBWEBSOCKETS_LIBRARIES})
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${LIBWEBSOCKETS_INCLUDE_DIR})


###############################################################################
# PTHREADS
###############################################################################
find_package (Threads MODULE REQUIRED)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# MPI LIB
################################################################################
find_package(MPI MODULE REQUIRED)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${MPI_C_LIBRARIES})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${MPI_CXX_LIBRARIES})

################################################################################
# IceT LIB
################################################################################
find_package (IceT MODULE REQUIRED)
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${ICET_CORE_LIBS})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${ICET_MPI_LIBS})
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${ICET_INCLUDE_DIRS})


################################################################################
# BOOST LIB
################################################################################
find_package(Boost 1.56.0 MODULE REQUIRED)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)


################################################################################
# Alpaka LIB
################################################################################
find_package(alpaka)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(ISAAC_LIBRARIES ${ISAAC_LIBRARIES} ${alpaka_LIBRARIES})
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} ${alpaka_DEFINITIONS})
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})


################################################################################
# CUDA LIB
################################################################################
find_package( CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})


if (ISAAC_CUDA)
endif(ISAAC_CUDA)

if (ISAAC_ALPAKA)
  set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_ALPAKA")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-literal-suffix" )
  list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
endif()
