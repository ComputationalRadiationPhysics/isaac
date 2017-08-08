cmake_minimum_required (VERSION 3.3)
project (example)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

set(ISAAC_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ISAAC_DIR})
set(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../alpaka/" CACHE STRING  "The location of the alpaka library")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")

find_package(ISAAC 1.1.0 REQUIRED)

set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_MAX_FUNCTORS=3")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_FUNCTOR_POW_ENABLED=0")

add_definitions(${ISAAC_DEFINITIONS})
include_directories(example_cuda ${ISAAC_INCLUDE_DIRS})

SET(ISAAC_STEREO "No" CACHE STRING "Using stereoscopy")
SET_PROPERTY(CACHE ISAAC_STEREO PROPERTY STRINGS No SideBySide Anaglyph)

if(${ISAAC_STEREO} STREQUAL "No")
	add_definitions(-DISAAC_STEREO=0)
endif()
if(${ISAAC_STEREO} STREQUAL "SideBySide")
	add_definitions(-DISAAC_STEREO=1)
endif()
if(${ISAAC_STEREO} STREQUAL "Anaglyph")
	add_definitions(-DISAAC_STEREO=2)
endif()


if (ISAAC_ALPAKA)
	ALPAKA_ADD_EXECUTABLE(isaac_example_alpaka ${CMAKE_SOURCE_DIR}/example.cpp)
	target_link_libraries(isaac_example_alpaka ${ISAAC_LIBRARIES})
	install(TARGETS isaac_example_alpaka DESTINATION bin )
endif()

if (ISAAC_CUDA)
	REMOVE_DEFINITIONS(REMOVE_ITEM "-DISAAC_ALPAKA")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11)
    list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
	SET_SOURCE_FILES_PROPERTIES(${CMAKE_SOURCE_DIR}/example.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	CUDA_ADD_EXECUTABLE(isaac_example_cuda ${CMAKE_SOURCE_DIR}/example.cpp)
	target_link_libraries(isaac_example_cuda ${ISAAC_LIBRARIES})
	install(TARGETS isaac_example_cuda DESTINATION bin )
endif()
