Sara: C++ Computer Vision Library
=================================

**I HAVE MOVED TO GITLAB, PLEASE GOTO: https://gitlab.com/DO-CV/sara**

**SO LONG GITHUB!**


[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5980a04834a04289a35752401d502728)](https://www.codacy.com/app/davidok8/sara?utm_source=github.com&utm_medium=referral&utm_content=DO-CV/sara&utm_campaign=badger)
![GitHub Actions Build Status](https://github.com/github/docs/actions/workflows/ci.yml/badge.svg?branch=master)
<a href="https://ci.appveyor.com/project/do-cv/sara"><img src="https://ci.appveyor.com/api/projects/status/github/do-cv/sara?branch=master&svg=true" alt="Build Status: Windows" /></a>
[![Travis Build Status](https://travis-ci.org/DO-CV/sara.svg?branch=master)](https://travis-ci.org/DO-CV/sara)
[![Coverage Status](https://coveralls.io/repos/DO-CV/sara/badge.svg?branch=master)](https://coveralls.io/r/DO-CV/sara?branch=master)
[![License](https://img.shields.io/badge/license-MPL2-blue.svg)](LICENSE)
[![Codedocs Reference API Documentation](https://codedocs.xyz/DO-CV/sara.svg)](https://codedocs.xyz/DO-CV/sara/)

*Sara* (सार) is a Sanskrit word meaning *essence*.

*Sara* focuses on:

1. having an **easy-to-use and simple API**,
2. having **easy-to-understand and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.

The design of *Sara* libraries is driven by the KISS principle. I enjoy
maintaining *Sara* and making it evolve when I feel like it.

*Sara* is licensed with the [Mozilla Public License version
2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).


**As of June 12, 2019, Sara is now a C++ 17 only project.** The improvements in
the language are compelling enough to make the library more readable.

**Supported compilers:**
- Visual Studio 2017 and above
- gcc 7 and above


Documentation
-------------

I don't have much time to maintain the documentation. **I'd love your help**.

What I can easily do is to keep up-to-date the [reference API
documentation](https://codedocs.xyz/DO-CV/sara/) hosted in **codedocs.xyz**.

I also write [documentation](https://sara-github.readthedocs.org/) at
**readthedocs.org** where I provide more mathematical details about my
implementations.

In any case you are always better off consulting the [examples
folder](https://github.com/DO-CV/sara/tree/master/cpp/examples) and the [test
folder](https://github.com/DO-CV/sara/tree/master/cpp/test).

The codes are generally short and carefully so they should help you to get up to
speed with the library usage.


Why yet another library?
------------------------

I am aware that *OpenCV* exists.

I simply like crafting software from the ground up and understanding Computer
Vision algorithms by reimplementing them from A to Z. Besides, not everybody in
the industry likes and uses *OpenCV*.

I used OpenCV C API for the first time during my research internship at Siemens
in 2008 and was not aware of its C++ API at all. It was a very frustrating
experience especially for the inexperienced programmer that I was back then.

After a while, I started writing the library as a hobby to have a more
easy-to-use library and also to gain a better mastery of the C++ language. Then
the library *Sara* became more structured in 2009, when I started my PhD at the
[IMAGINE lab](http://imagine.enpc.fr/) in [Ecole des Ponts,
ParisTech](http://www.enpc.fr).

Parts of the library may be reused for applications in the industry as I do
myself. The library is not designed for real-time critical applications and you
should use **OpenCV** (among others) instead. In most use cases, the library
should be fine otherwise.

Time has flown. Years after years I try to keep the library alive but things
happen a lot slowly.


Build the libraries
-------------------

**The information below is a bit outdated but the detailed steps are still
useful. Your best bet is to have a look at the CI scripts like `.gitlab-ci.yml`
or `.travis.yml`**

To build the libraries, run:

1. Install the following packages:

   - On Debian-based distributions:
     ```
     sudo apt-get install -qq \
       cmake \
       doxygen \
       libjpeg8-dev \
       libpng12-dev \
       libtiff5-dev \
       libavcodec-ffmpeg-dev \
       libavformat-ffmpeg-dev \
       libavutil-ffmpeg-dev \
       qtbase5-dev

     # To install Python bindings.
     sudo apt-get install -qq \
       boost-python-dev \
       python3-dev
     ```

   - On Red Hat-based distributions:
     ```
     sudo yum install -y
       cmake \
       doxygen \
       libboost-test-dev \
       libjpeg-devel \
       libpng-devel \
       libtiff-devel \
       ffmpeg \
       ffmpeg-devel \
       qt-devel

     # To install Python bindings.
     sudo apt-get install -qq \
       libboost-python-dev \
       libpython3-devel
     ```

2. Build the library:

   ```
   mkdir build
   cd build
   cmake .. \
     -DCMAKE_BUILD_TYPE=Release \
     -DSARA_BUILD_SHARED_LIBS=ON \
     -DSARA_BUILD_SAMPLES=ON \
     -DSARA_BUILD_TESTS=ON
   make  -j`nproc`  # to build with all your CPU cores.
   ```

3. Run the tests to make sure everything is alright.

   ```
   ctest --output-on-failure
   ```

4. Create DEB and RPM package.

   ```
   make package
   ```

5. Deploy by install the Debian package with Ubuntu Software Center, or type:

   ```
   # Debian-based distros:
   sudo dpkg -i libDO-Sara-shared-{version}.deb

   # Red Hat-based distros:
   sudo rpm -i libDO-Sara-shared-{version}.deb
   ```
