cmake_minimum_required (VERSION 3.3)
project (example)

set(ISAAC_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ISAAC_DIR})
set(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../requirement-modules/alpaka/" CACHE STRING  "The location of the alpaka library")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ALPAKA_ROOT}")

find_package(ISAAC 0.1.0 REQUIRED)

set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_MAX_FUNCTORS=3")
set(ISAAC_DEFINITIONS ${ISAAC_DEFINITIONS} "-DISAAC_FUNCTOR_POW_ENABLED=0")

add_definitions(${ISAAC_DEFINITIONS})
include_directories(example_cuda ${ISAAC_INCLUDE_DIRS})

if (ISAAC_ALPAKA)
	ALPAKA_ADD_EXECUTABLE(example_alpaka ${CMAKE_SOURCE_DIR}/example.cu)
	target_link_libraries(example_alpaka ${ISAAC_LIBRARIES})
endif()

if (ISAAC_CUDA)
	REMOVE_DEFINITIONS(REMOVE_ITEM "-DISAAC_ALPAKA")
	CUDA_ADD_EXECUTABLE(example_cuda ${CMAKE_SOURCE_DIR}/example.cu)
	target_link_libraries(example_cuda ${ISAAC_LIBRARIES})
endif()
