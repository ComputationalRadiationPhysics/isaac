cmake_minimum_required (VERSION 2.6)
project (isaac)

set(SRCS
	${CMAKE_SOURCE_DIR}/server/isaac.cpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.cpp
	${CMAKE_SOURCE_DIR}/server/Master.cpp
)

set(HDRS
	${CMAKE_SOURCE_DIR}/server/Common.hpp
	${CMAKE_SOURCE_DIR}/server/MetaDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/WebSocketDataConnector.hpp
	${CMAKE_SOURCE_DIR}/server/Master.hpp
	${CMAKE_SOURCE_DIR}/server/ThreadList.hpp
)

add_executable(isaac ${HDRS} ${SRCS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")
find_package(LibWebSockets REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})

find_package (Threads)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(isaac ${LIBS})
