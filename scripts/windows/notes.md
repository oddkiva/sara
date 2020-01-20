Third-Party Software on Windows
===============================

**Short answer**: use **vcpkg** whenever possible!

Otherwise we can rebuild things ourselves as follows:

- **Qt5**: if we need to rebuild from sources:
  - Download `jom` and put it in the source folder.
  - Run `./configure.bat`
  - I'd rather not use `vcpkg` for Qt5.
- **Boost**: use and readapt the configuration `user-config.jam` to build
  Boost binaries with `bjam`.
  - `vcpkg` does not bring much value to build boost.
- **HDF5**:
  - Yes: prefer reusing `vcpkg`
  - Not recommended but alternatively: we can download the sources from the
    website and change the **CMake** config file to build for VS 2019.
- **FFmpeg**: prefer reusing `vcpkg`. Need to remove the binaries...
- **Image I/O**: prefer reusing `vcpkg`. Keep the sources just in case because
  they are quick to build.
- **Ceres Solver**: prefer reusing `vcpkg`. Keep the sources just in
  case.
