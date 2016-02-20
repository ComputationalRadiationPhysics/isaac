# - Config file for the isaac package
# It defines the following variables
#  isaac_INCLUDE_DIRS - include directories for FooBar
#  isaac_LIBRARIES    - libraries to link against
#  isaac_DEFINITIONS  - necessary definitions
#
# It defines the following options
#  ISAAC_THREADING
#  ISAAC_SHOWBORDER
#  ISAAC_CUDA
#  ISAAC_ALPAKA
#  ISAAC_JPEG

###############################################################################
# isaac
###############################################################################
cmake_minimum_required (VERSION 2.6)
cmake_policy(SET CMP0048 OLD)
project (isaac)

set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} "${isaac_DIR}/insitu")
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} "${isaac_DIR}/insitu/isaac")

set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-std=c++11")
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-march=native")
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-mtune=native")
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-DISAAC_MAX_FUNCTORS=3")
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-DISAAC_FUNCTOR_POW_ENABLED=0")

###############################################################################
# MODULES
###############################################################################
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${isaac_DIR}/Modules")


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
	set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${JPEG_INCLUDE_DIR})
	set(isaac_LIBRARIES ${isaac_LIBRARIES} ${JPEG_LIBRARY})
	set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-DISAAC_JPEG")
endif (ISAAC_JPEG)


###############################################################################
# PKGCONFIG 
###############################################################################
find_package (PkgConfig MODULE REQUIRED)


###############################################################################
# JANSSON LIB
###############################################################################
find_package (Jansson MODULE REQUIRED)
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${JANSSON_LIBRARIES})
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${JANSSON_INCLUDE_DIRS})


###############################################################################
# LIBWEBSOCKETS
###############################################################################
find_package(LibWebSockets MODULE REQUIRED)
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${LIBWEBSOCKETS_LIBRARIES})
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${LIBWEBSOCKETS_INCLUDE_DIR})


###############################################################################
# PTHREADS
###############################################################################
find_package (Threads MODULE REQUIRED)
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})


################################################################################
# MPI LIB
################################################################################
find_package(MPI MODULE REQUIRED)
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${MPI_C_LIBRARIES})
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${MPI_CXX_LIBRARIES})

################################################################################
# IceT LIB
################################################################################
find_package (IceT MODULE REQUIRED)
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${ICET_CORE_LIBS})
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${ICET_MPI_LIBS})
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${ICET_INCLUDE_DIRS})


################################################################################
# BOOST LIB
################################################################################
find_package(Boost 1.56.0 MODULE REQUIRED)
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)


################################################################################
# Alpaka LIB
################################################################################
find_package(alpaka)
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(isaac_LIBRARIES ${isaac_LIBRARIES} ${alpaka_LIBRARIES})
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} ${alpaka_DEFINITIONS})
set(isaac_DEFINITIONS ${isaac_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

################################################################################
# CUDA LIB
################################################################################
find_package( CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
set(isaac_INCLUDE_DIRS ${isaac_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})


if (ISAAC_CUDA)
endif(ISAAC_CUDA)

if (ISAAC_ALPAKA)
  set(isaac_DEFINITIONS ${isaac_DEFINITIONS} "-DISAAC_ALPAKA")          
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-literal-suffix" )
  list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
endif()

