cmake_minimum_required (VERSION 3.3)
project (isaac)

set(SRCS
	${CMAKE_SOURCE_DIR}/server/isaac.cpp
	${CMAKE_SOURCE_DIR}/server/Common.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/ImageConnector.cpp
	${CMAKE_SOURCE_DIR}/server/InsituConnector.cpp
	${CMAKE_SOURCE_DIR}/server/InsituConnectorMaster.cpp
	${CMAKE_SOURCE_DIR}/server/Broker.cpp
	${CMAKE_SOURCE_DIR}/server/Runable.cpp
	${CMAKE_SOURCE_DIR}/server/MessageAble.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataClient.cpp
	${CMAKE_SOURCE_DIR}/server/TCPDataConnector.cpp
)

set(HDRS
	${CMAKE_SOURCE_DIR}/server/Common.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/ImageConnector.hpp
	${CMAKE_SOURCE_DIR}/server/InsituConnector.hpp
	${CMAKE_SOURCE_DIR}/server/InsituConnectorMaster.hpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/Broker.hpp
	${CMAKE_SOURCE_DIR}/server/ThreadList.hpp
	${CMAKE_SOURCE_DIR}/server/Runable.hpp
	${CMAKE_SOURCE_DIR}/server/MessageAble.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataClient.hpp
	${CMAKE_SOURCE_DIR}/server/TCPDataConnector.hpp
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
	set(SRCS ${SRCS} ${CMAKE_SOURCE_DIR}/server/RTMPImageConnector.cpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/server/RTPImageConnector.hpp)
	set(HDRS ${HDRS} ${CMAKE_SOURCE_DIR}/server/RTMPImageConnector.hpp)

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

add_definitions(-std=c++11 -march=native -mtune=native)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")

find_package (Jansson CONFIG REQUIRED)
set(LIBS ${LIBS} ${JANSSON_LIBRARIES})
set(INCLUDE ${INCLUDE} ${JANSSON_INCLUDE_DIRS})

find_package(Libwebsockets CONFIG REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})
set(INCLUDE ${INCLUDE} ${LIBWEBSOCKETS_INCLUDE_DIRS})

find_package (Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

include_directories(isaac ${INCLUDE})	
target_link_libraries(isaac ${LIBS})
