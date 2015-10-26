cmake_minimum_required (VERSION 2.6)
project (isaac)

set(SRCS
	${CMAKE_SOURCE_DIR}/server/isaac.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.cpp
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
	${CMAKE_SOURCE_DIR}/server/InsituConnector.hpp
	${CMAKE_SOURCE_DIR}/server/InsituConnectorMaster.hpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/Master.hpp
	${CMAKE_SOURCE_DIR}/server/ThreadList.hpp
	${CMAKE_SOURCE_DIR}/server/Runable.hpp
	${CMAKE_SOURCE_DIR}/server/MessageAble.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataClient.hpp
)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package(LibWebSockets REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})
set(INCLUDE ${INCLUDE} ${LIBWEBSOCKETS_INCLUDE_DIR})

find_package (Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package (Jansson)
set(LIBS ${LIBS} ${JANSSON_LIBRARIES} REQUIRED)
set(INCLUDE ${INCLUDE} ${JANSSON_INCLUDE_DIRS})

include_directories(isaac ${INCLUDE})	
target_link_libraries(isaac ${LIBS})

option(BUILD_EXAMPLE "Build the example." ON)

if (BUILD_EXAMPLE)
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
	
	set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/requirement-modules/alpaka/" CACHE STRING  "The location of the alpaka library")
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")
	find_package( alpaka  REQUIRED)
	
	set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-literal-suffix" )

	include_directories(example ${EXAMPLE_INCLUDE} ${alpaka_INCLUDE_DIRS})	
	add_definitions( ${alpaka_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
	ALPAKA_ADD_EXECUTABLE(example ${CMAKE_SOURCE_DIR}/insitu/isaac.hpp ${CMAKE_SOURCE_DIR}/insitu/example.cpp)
	target_link_libraries(example ${EXAMPLE_LIBS})
endif ()
