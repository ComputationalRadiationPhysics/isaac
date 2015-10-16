cmake_minimum_required (VERSION 2.6)
project (isaac)
add_executable(isaac server/server.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules")
find_package(LibWebSockets REQUIRED)
set(LIBS ${LIBS} ${LIBWEBSOCKETS_LIBRARIES})

target_link_libraries(isaac ${LIBS})
