# Changelog

## [v1.5.2] - 2021-01-20

alpaka 0.6.0 is required if you enabled alpaka support.

* Server
  - fix jansson 2.12+ support #123
* Library
  - fix missing include #124
  - fix jansson 2.12+ support #123
  - apply namespace refactoring from alpaka 0.6.0rc3 #122


## [v1.5.1] - 2020-11-06

* Server
  - Send/Receive buffer sizes adjustable in cmake as ISAAC_MAX_RECEIVE + added warnings/errors on overflow #115
  - Fix for newer LibWebSocket versions #108
  - Replaced imageConnectorList with map to guarantee correct ID #106

* Library
  - enable AO bug via pre-compiler flag #112
  - Fix for wrong depth values on clipping plane for ambient occlusion calculations #111
  - Use correct MIN/MAX functions in kernel #109
  - fix compile error (missing include) #107


## [v1.5.0] - 2020-05-04

The versioning of the server follows now the library versioning.

- CMake: remove manipulation of CUDA_NVCC_FLAGS #101
- Add Simple ambient occlusion effect #98
- Modern browser fix and client code reorganization #92
- Alpaka example fix + Updated Alpaka function calls #89
- Particle integration in library and example #88
- Server fixes #87


## [v1.4.0] - 2018-06-12

- Fixed broken iso surface calculation for border regions
- Fixed visible egdes between node borders in iso surface rendering
- Fixed memory access of gradient calculation in two-cell border around volume
- Added simple ortho projection functionality
- Fixed installed config path


## [v1.3.3] - 2017-11-03

- removed broken and useless(?) include of a header file not available anymore in boost version 1.65.0 and above


## [v1.3.2] - 2017-11-03

- set needed libwebsockets version to 2.1.1 or above as 2.10 is buggy
- fixed broken CUDACC_VER of CUDA 9
- fixed build for building without CUDA
- added a define ISAAC_IDX_TYPE for changing the index type when using Alpaka
- removed isaac_uint_dim as it was not used and made problems
- renamed the internal variable threadIdx as the nvcc may confuse it with it's own
- fixed some GCC 7.1 warnings
- using boolean definitions from libjpeg headers
- fixed ISAAC_HOST_DEVICE_INLINE macro
- added manual control buttons to the examples


## [v1.3.1] - 2017-07-06

- Fixed version macros


## [v1.3.0] - 2017-04-06

- Fixed ISAAC CMake Config and installation
- Fixed bug in server, which prevented more than one connection in some cases


## [v1.2.0] - 2017-01-05

- Adding protocol string (major and minor)
- Splitted server and library version (both major, minor and patch)
- Splitted server and library CMake files
- Some little bug fixing


## [v1.1.0] - 2016-11-10


## [v1.0.0] - 2016-05-26
