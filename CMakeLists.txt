cmake_minimum_required (VERSION 2.6)
project (isaac)

set(SRCS
	${CMAKE_SOURCE_DIR}/server/isaac.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/ImageConnector.cpp
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
endif (ISAAC_SDL)

find_package (PkgConfig REQUIRED)

option(ISAAC_GST "Use Gstreamer" ON)
if (ISAAC_GST)
	pkg_search_module(GST REQUIRED gstreamer-1.0)
	set(LIBS ${LIBS} ${GST_LIBRARIES} )
	set(INCLUDE ${INCLUDE} ${GST_INCLUDE_DIRS})

	pkg_search_module(GST_APP REQUIRED gstreamer-app-1.0)
	set(LIBS ${LIBS} ${GST_APP_LIBRARIES})
	set(INCLUDE ${INCLUDE} ${GST_APP_INCLUDE_DIRS})

	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/server/RTPImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/server/RTPImageConnector.hpp)

	add_definitions(-DISAAC_GST)
endif (ISAAC_GST)

option(ISAAC_JPEG "Use JPEG compression between visualization and isaac server. Deactivating will not work with big images. And with big I am talking about bigger than 800x600." ON)
if (ISAAC_JPEG)
	find_package(JPEG REQUIRED)
	set(INCLUDE ${INCLUDE} ${JPEG_INCLUDE_DIR})
	set(LIBS ${LIBS} ${JPEG_LIBRARY})

	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/server/URIImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/server/URIImageConnector.hpp)

	add_definitions(-DISAAC_JPEG)
endif (ISAAC_JPEG)

add_definitions(-std=c++11)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Jansson REQUIRED)
set(LIBS ${LIBS} ${JANSSON_LIBRARIES})
set(INCLUDE ${INCLUDE} ${JANSSON_INCLUDE_DIRS})

find_package(LibWebSockets REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})
set(INCLUDE ${INCLUDE} ${LIBWEBSOCKETS_INCLUDE_DIR})

find_package (Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

include_directories(isaac ${INCLUDE})	
target_link_libraries(isaac ${LIBS})

option(BUILD_EXAMPLE_CUDA "Build the example using cuda." ON)
option(BUILD_EXAMPLE_ALPAKA "Build the example using alpaka." OFF)


if (BUILD_EXAMPLE_CUDA OR BUILD_EXAMPLE_ALPAKA)
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${JANSSON_INCLUDE_DIRS})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${JPEG_INCLUDE_DIR})

	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${CMAKE_THREAD_LIBS_INIT})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${JANSSON_LIBRARIES})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${JPEG_LIBRARY})

	find_package (IceT REQUIRED)
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${ICET_CORE_LIBS})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${ICET_MPI_LIBS})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${ICET_INCLUDE_DIRS})

	find_package (MPI REQUIRED)
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${MPI_C_LIBRARIES})
	set(EXAMPLE_LIBS ${EXAMPLE_LIBS} ${MPI_CXX_LIBRARIES})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${MPI_C_INCLUDE_PATH})
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${MPI_CXX_INCLUDE_PATH})

	find_package(Boost 1.56.0 REQUIRED)
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${Boost_INCLUDE_DIR})
	add_definitions(-DBOOST_ALL_NO_LIB)

	option(ISAAC_THREADING "Do the video and metadata transmission transport in background. May be buggy if not used with MPI_threading intialisation!" OFF)
	if (ISAAC_THREADING)
		add_definitions(-DISAAC_THREADING)
	endif ()
	
	set(EXAMPLE_INCLUDE ${EXAMPLE_INCLUDE} ${CMAKE_SOURCE_DIR}/insitu)
	
	add_definitions(-DISAAC_MAX_FUNCTORS=2)
	
	if (BUILD_EXAMPLE_CUDA)
		find_package( CUDA REQUIRED)
		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
		include_directories(example_cuda ${EXAMPLE_INCLUDE} ${CUDA_INCLUDE_DIRS})
		CUDA_ADD_EXECUTABLE(example_cuda ${CMAKE_SOURCE_DIR}/insitu/example.cu)
		target_link_libraries(example_cuda ${EXAMPLE_LIBS} ${CUDA_LIBRARIES})
	endif(BUILD_EXAMPLE_CUDA)
	
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
	endif(BUILD_EXAMPLE_ALPAKA)
endif (BUILD_EXAMPLE_CUDA OR BUILD_EXAMPLE_ALPAKA)
