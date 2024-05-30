Third-Party Software on Windows
===============================

**Short answer**: use **vcpkg** whenever possible!

Otherwise we can rebuild things ourselves as follows:

- **Qt5**: if we need to rebuild from sources:
  - Download `jom` and put it in the source folder.
  - Run `./configure.bat`
  - I'd rather not use `vcpkg` for Qt5.
- Halide: *DON'T* install from `vcpkg`. Grab the binaries instead from GitHub.
- **Boost**: use and readapt the configuration `user-config.jam` to build
  Boost binaries with `bjam`.
  - `vcpkg` does not bring much value to build boost.
- **HDF5**:
  - Yes: prefer reusing `vcpkg`
  - Not recommended but alternatively: we can download the sources from the
    website and change the **CMake** config file to build for VS 2019.
- **FFmpeg**: prefer reusing `vcpkg`.
- **Image I/O**: prefer reusing `vcpkg`.
- **Ceres Solver**: prefer reusing `vcpkg`.

## Supporting CUDA

Note to self: this is becoming very tricky and sometimes discouraging to
continue providing support on Windows...

As of May 30, 2024:
Visual Studio latest release is version 17.10.1. And CUDA 12.x does not support
version >= 17.10. Upon upgrading VS, CMake won't be able to find a valid host
CUDA compiler.

### 1. Roll back to  version: (v14.39-17.9)
To fix the problem, launch "Visual Studio Installer", reinstall the individual
components "MSVC v143 - VS 2022 C++ x64-x86 build tools (v14.39-17.9)".

Please pay *extra extra extra* attention to the version number: **v14.39-17.9**.

### 2. CMake dark magic: toolset version.

Specify the toolset version in the command line when composing the already
complicated CMake command line:

```
"cmake -S `"..\sara`" -B `".`" -G `"Visual Studio 2022 17`" -T `v143`"
```

The important bit in the command line is **the toolset version number**
`-T v143` so that we can compile CUDA code.

When NVIDIA releases a newer CUDA toolkit supports later versions of VS, this
option won't be necessary.

Otherwise keep readapting this, this should work until your dearly cherished
GPU is deemed obsolete by NVIDIA, so that they force you to sell a kidney, just
to buy a newer GPU...
