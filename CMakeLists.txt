cmake_minimum_required (VERSION 2.6)
project (isaac)

set(SRCS
	${CMAKE_SOURCE_DIR}/server/isaac.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/ImageConnector.cpp
	${CMAKE_SOURCE_DIR}/server/RTPImageConnector.cpp
	${CMAKE_SOURCE_DIR}/server/InsituConnector.cpp
	${CMAKE_SOURCE_DIR}/server/InsituConnectorMaster.cpp
	${CMAKE_SOURCE_DIR}/server/Master.cpp
	${CMAKE_SOURCE_DIR}/server/Runable.cpp
	${CMAKE_SOURCE_DIR}/server/MessageAble.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataClient.cpp
)

set(HDRS
	${CMAKE_SOURCE_DIR}/server/Common.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/ImageConnector.hpp
	${CMAKE_SOURCE_DIR}/server/RTPImageConnector.hpp
	${CMAKE_SOURCE_DIR}/server/InsituConnector.hpp
	${CMAKE_SOURCE_DIR}/server/InsituConnectorMaster.hpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/Master.hpp
	${CMAKE_SOURCE_DIR}/server/ThreadList.hpp
	${CMAKE_SOURCE_DIR}/server/Runable.hpp
	${CMAKE_SOURCE_DIR}/server/MessageAble.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataClient.hpp
)

option(ISAAC_SDL "Build SDL ImageConnector" OFF)
if (ISAAC_SDL)
	find_package(SDL REQUIRED)
	set(INCLUDE ${INCLUDE} ${SDL_INCLUDE_DIR})
	set(LIBS ${LIBS} ${SDL_LIBRARY})
	add_definitions(-DISAAC_SDL)
	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/server/SDLImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/server/SDLImageConnector.hpp)
endif ()

add_definitions(-std=c++11)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (PkgConfig REQUIRED)

find_package (Jansson REQUIRED)
set(LIBS ${LIBS} ${JANSSON_LIBRARIES})
set(INCLUDE ${INCLUDE} ${JANSSON_INCLUDE_DIRS})

find_package(LibWebSockets REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})
set(INCLUDE ${INCLUDE} ${LIBWEBSOCKETS_INCLUDE_DIR})

find_package (Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

#find_package (GStreamer REQUIRED)
pkg_search_module(GSTREAMER REQUIRED gstreamer-1.0)
set(LIBS ${LIBS} ${GSTREAMER_LIBRARIES} ${GSTREAMER_APP_LIBRARIES})
set(INCLUDE ${INCLUDE} ${GSTREAMER_INCLUDE_DIRS})

include_directories(isaac ${INCLUDE})	
target_link_libraries(isaac ${LIBS})

option(BUILD_EXAMPLE_CUDA "Build the example using cuda." ON)
option(BUILD_EXAMPLE_ALPAKA "Build the example using alpaka." OFF)


if (BUILD_EXAMPLE_CUDA OR BUILD_EXAMPLE_ALPAKA)
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${CMAKE_THREAD_LIBS_INIT})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${JANSSON_LIBRARIES})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${JANSSON_INCLUDE_DIRS})

	find_package (IceT REQUIRED)
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${ICET_CORE_LIBS})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${ICET_MPI_LIBS})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${ICET_INCLUDE_DIRS})

	find_package (MPI REQUIRED)
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${MPI_C_LIBRARIES})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${MPI_CXX_LIBRARIES})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${MPI_C_INCLUDE_PATH})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${MPI_CXX_INCLUDE_PATH})

	find_package(Boost REQUIRED)
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${Boost_INCLUDE_DIR})
	add_definitions(-DBOOST_ALL_NO_LIB)

	option(ISAAC_THREADING "Do the vizualisation in background. May be buggy. Needs MPI_threading intialisation!" OFF)
	if (ISAAC_THREADING)
		add_definitions(-DISAAC_THREADING)
	endif ()
	
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${CMAKE_SOURCE_DIR}/insitu)
	
	if (BUILD_EXAMPLE_CUDA)
		find_package( CUDA REQUIRED)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
		include_directories(example_cuda ${EXAMPLE_INCLUDE} ${CUDA_INCLUDE_DIRS})
		CUDA_ADD_EXECUTABLE(example_cuda ${CMAKE_SOURCE_DIR}/insitu/example.cu)
		target_link_libraries(example_cuda ${EXAMPLE_LIBS} ${CUDA_LIBRARIES})
	endif()
	
	if (BUILD_EXAMPLE_ALPAKA)
		set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/requirement-modules/alpaka/" CACHE STRING  "The location of the alpaka library")
		set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")
		find_package( alpaka  REQUIRED)
		set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-literal-suffix" )
		include_directories(example_alpaka ${EXAMPLE_INCLUDE} ${alpaka_INCLUDE_DIRS})	
		add_definitions( ${alpaka_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
		add_definitions(-DISAAC_ALPAKA)
		list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
		ALPAKA_ADD_EXECUTABLE(example_alpaka ${CMAKE_SOURCE_DIR}/insitu/isaac.hpp ${CMAKE_SOURCE_DIR}/insitu/example.cu)
		target_link_libraries(example_alpaka ${EXAMPLE_LIBS})
	endif()
endif ()
