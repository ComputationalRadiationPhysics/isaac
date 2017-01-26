cmake_minimum_required (VERSION 3.3)
project (isaac)

install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/isaac.hpp" DESTINATION "include" )
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/isaac" DESTINATION "include" PATTERN "mainpage.dox" EXCLUDE )
install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/ISAACConfig.cmake" DESTINATION "lib/cmake/ISAAC" )
install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/ISAACConfigVersion.cmake" DESTINATION "lib/cmake/ISAAC" )
