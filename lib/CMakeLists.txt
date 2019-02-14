cmake_minimum_required (VERSION 3.3)
project (isaac)

# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

configure_file (
	"${PROJECT_SOURCE_DIR}/ISAACConfigVersion.cmake.in"
	"${PROJECT_BINARY_DIR}/ISAACConfigVersion.cmake"
	@ONLY
	)
configure_file (
    "${PROJECT_SOURCE_DIR}/ISAACBaseDir.cmake.in"
    "${PROJECT_BINARY_DIR}/ISAACBaseDir.cmake"
    @ONLY
    )

install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/isaac.hpp" DESTINATION "include" )
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/isaac" DESTINATION "include" PATTERN "mainpage.dox" EXCLUDE )
install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/ISAACConfig.cmake" DESTINATION "lib/cmake/ISAAC" )
install( FILES "${PROJECT_BINARY_DIR}/ISAACConfigVersion.cmake" DESTINATION "lib/cmake/ISAAC" )
install( FILES "${PROJECT_BINARY_DIR}/ISAACBaseDir.cmake" DESTINATION "lib/cmake/ISAAC" )
install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Modules" DESTINATION "lib/cmake/ISAAC")
