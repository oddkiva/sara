Sara: C++ Computer Vision Library
=================================

**I HAVE MOVED TO GITLAB, PLEASE GOTO: https://gitlab.com/DO-CV/sara**

**SO LONG GITHUB!**


[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5980a04834a04289a35752401d502728)](https://www.codacy.com/app/davidok8/sara?utm_source=github.com&utm_medium=referral&utm_content=DO-CV/sara&utm_campaign=badger)
[![Build Status](https://travis-ci.org/DO-CV/sara.svg?branch=master)](https://travis-ci.org/DO-CV/sara)
<a href="https://ci.appveyor.com/project/do-cv/sara"><img src="https://ci.appveyor.com/api/projects/status/github/do-cv/sara?branch=master&svg=true" alt="Build Status: Windows" /></a>
[![Coverage Status](https://coveralls.io/repos/DO-CV/sara/badge.svg?branch=master)](https://coveralls.io/r/DO-CV/sara?branch=master)
[![License](https://img.shields.io/badge/license-MPL2-blue.svg)](LICENSE)

*Sara* (सार) is a Sanskrit word meaning *essence*.

*Sara* focuses on:

1. having an **easy-to-use and simple API**,
2. having **easy-to-understand and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.

The design of *Sara* libraries is driven by the KISS principle.

Try it and feel free to become involved with the development of the libraries.

I dedicate lots of patience and love to maintain *Sara* and make it evolve as
much as my time and energy allow it.

*Sara* is licensed with the [Mozilla Public License version
2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).


**As of June 12, 2019, Sara is now a C++ 17 only project.** The small
improvements in the language made my life much easier and the code more
pythonic.

**Supported compilers:**
- Visual Studio 2017 and above
- gcc 7 and above


Documentation
-------------

I don't have much time to maintain the documentation. **I'd love your help**.

What I can easily do is to keep up-to-date the reference documentation
[here](http://do-cv.github.io/sara/). There is also some more friendly
documentation on at the [readthedocs.org](https://readthedocs.org/) but it is
not up-to-date.

Honestly you will be much better off consulting the [examples
folder](https://github.com/DO-CV/sara/tree/master/cpp/examples) and the [test
folder](https://github.com/DO-CV/sara/tree/master/cpp/test).

The codes are generally short and carefully so they should help you to get up to
speed with the library usage.


Why yet another library?
------------------------

I started writing *Sara* in 2009, when I started my PhD at the [IMAGINE
lab](http://imagine.enpc.fr/) in [Ecole des Ponts,
ParisTech](http://www.enpc.fr).

Historically, I started writing DO-CV before openCV came up with a new C++ API
(In late 2015, a computer vision researcher was shocked when I told him that I
don't like openCV and told me patronizingly that the openCV C++ API was actually
released in 2007. Well I did not know about computer vision yet. Anyways so
what?)

I used openCV for the first time during my research internship at Siemens. That
was in 2008 (quite some time, now that I think of it!). I was very frustrated
with it. After a while, I started writing the library as a hobby to have a more
easy-to-use library and also to gain a better mastery of the C++ language. Now,
the library keeps evolving and can be reused for serious applications in the
industry.

Today openCV has evolved a lot. Despite that openCV has yet to convince me to
use it, API-wise. Besides, not everybody in the industry uses *openCV*.

I like my library and it is still *alive*, *lightweight*, *tested* since 2009!


Build the libraries
-------------------

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
