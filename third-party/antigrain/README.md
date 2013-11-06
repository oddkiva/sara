Anti-grain Geometry
===================

The original **Anti-grain Geometry (AGG)** project was a creation of **Maxim Shemanarev** back in 2006.

For further information please visit http://www.antigrain.com


## Important Note
This forked repository hosts a CMake build only. Get CMake (http://www.cmake.org) to generate projects files or makefile. I have deliberately removed autotools and makefile. Sorry about that, if you really wanted them.

Supported compilers:
* VS 2010
* VS 2012
* Clang 5.0.0

This branch "Core" contains only the core source files needed to do drawing with AGG on a memory buffer. There is no support for font drawing.
