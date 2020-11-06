cmake_minimum_required (VERSION 3.3)
project (isaac)

# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(SRCS
	${CMAKE_SOURCE_DIR}/src/isaac.cpp
	${CMAKE_SOURCE_DIR}/src/Common.cpp
	${CMAKE_SOURCE_DIR}/src/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/src/WebSocketDataConnector.cpp
	${CMAKE_SOURCE_DIR}/src/ImageConnector.cpp
	${CMAKE_SOURCE_DIR}/src/InsituConnector.cpp
	${CMAKE_SOURCE_DIR}/src/InsituConnectorMaster.cpp
	${CMAKE_SOURCE_DIR}/src/Broker.cpp
	${CMAKE_SOURCE_DIR}/src/Runable.cpp
	${CMAKE_SOURCE_DIR}/src/MessageAble.cpp
	${CMAKE_SOURCE_DIR}/src/MetaDataClient.cpp
	${CMAKE_SOURCE_DIR}/src/TCPDataConnector.cpp
	${CMAKE_SOURCE_DIR}/src/NetworkInterfaces.cpp
	${CMAKE_SOURCE_DIR}/src/SaveFileImageConnector.cpp
)

set(HDRS
	${CMAKE_SOURCE_DIR}/src/Common.hpp
	${CMAKE_SOURCE_DIR}/src/MetaDataConnector.hpp
	${CMAKE_SOURCE_DIR}/src/ImageConnector.hpp
	${CMAKE_SOURCE_DIR}/src/InsituConnector.hpp
	${CMAKE_SOURCE_DIR}/src/InsituConnectorMaster.hpp
	${CMAKE_SOURCE_DIR}/src/WebSocketDataConnector.hpp
	${CMAKE_SOURCE_DIR}/src/Broker.hpp
	${CMAKE_SOURCE_DIR}/src/ThreadList.hpp
	${CMAKE_SOURCE_DIR}/src/Runable.hpp
	${CMAKE_SOURCE_DIR}/src/MessageAble.hpp
	${CMAKE_SOURCE_DIR}/src/MetaDataClient.hpp
	${CMAKE_SOURCE_DIR}/src/TCPDataConnector.hpp
	${CMAKE_SOURCE_DIR}/src/NetworkInterfaces.hpp
	${CMAKE_SOURCE_DIR}/src/SaveFileImageConnector.hpp
	${CMAKE_SOURCE_DIR}/src/version.hpp
)

option(ISAAC_SDL "Build SDL ImageConnector" OFF)
if (ISAAC_SDL)
	find_package(SDL REQUIRED)
	set(INCLUDE ${INCLUDE} ${SDL_INCLUDE_DIR})
	set(LIBS ${LIBS} ${SDL_LIBRARY})
	add_definitions(-DISAAC_SDL)
	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/src/SDLImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/src/SDLImageConnector.hpp)
endif (ISAAC_SDL)

find_package (PkgConfig REQUIRED)

set(GST_INITIAL_OPTION OFF)
pkg_search_module(GST gstreamer-1.0)
if (GST_FOUND)
	set(GST_INITIAL_OPTION ON)
endif(GST_FOUND)

option(ISAAC_GST "Use Gstreamer" ${GST_INITIAL_OPTION})
if (ISAAC_GST)
	pkg_search_module(GST REQUIRED gstreamer-1.0)
	set(LIBS ${LIBS} ${GST_LIBRARIES} )
	set(INCLUDE ${INCLUDE} ${GST_INCLUDE_DIRS})

	pkg_search_module(GST_APP REQUIRED gstreamer-app-1.0)
	set(LIBS ${LIBS} ${GST_APP_LIBRARIES})
	set(INCLUDE ${INCLUDE} ${GST_APP_INCLUDE_DIRS})

	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/src/RTPImageConnector.cpp)
	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/src/RTMPImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/src/RTPImageConnector.hpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/src/RTMPImageConnector.hpp)

	add_definitions(-DISAAC_GST)
endif (ISAAC_GST)



set(ISAAC_MAX_RECEIVE "8388608" CACHE STRING "The size of the reserved buffers for sending and receiving, default of 8 MB should work for up to 4K" )
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} -DISAAC_MAX_RECEIVE=${ISAAC_MAX_RECEIVE})
add_definitions(${ISAAC_DEFINITIONS})

option(ISAAC_JPEG "Use JPEG compression between visualization and isaac server. Deactivating will not work with big images. And with big I am talking about bigger than 800x600." ON)
if (ISAAC_JPEG)
	find_package(JPEG REQUIRED)
	set(INCLUDE ${INCLUDE} ${JPEG_INCLUDE_DIR})
	set(LIBS ${LIBS} ${JPEG_LIBRARY})

	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/src/URIImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/src/URIImageConnector.hpp)

	add_definitions(-DISAAC_JPEG)
endif (ISAAC_JPEG)

add_definitions(-std=c++11 -march=native -mtune=native)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Jansson CONFIG REQUIRED)
set(LIBS ${LIBS} ${JANSSON_LIBRARIES})
set(INCLUDE ${INCLUDE} ${JANSSON_INCLUDE_DIRS})

find_package(Libwebsockets 2.1.1 CONFIG REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})
set(INCLUDE ${INCLUDE} ${LIBWEBSOCKETS_INCLUDE_DIRS})

find_package (Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(Boost MODULE REQUIRED)
set(INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)

include_directories(isaac ${INCLUDE})
target_link_libraries(isaac ${LIBS})

install( TARGETS isaac DESTINATION bin )
