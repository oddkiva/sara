DO-CV: C++ Computer Vision Library
==================================

DO-CV focuses on:

1. having an **easy-to-use and simple API**,
2. having **easy-to-understand and efficient** implementations of computer vision
   algorithms,
3. **rigorous testing**.

Try it and feel free to become involved with the development of the libraries.

You can rest assured that I dedicate lots of love to maintain DO-CV and make it
evolve as much as my time and energy allow it.

DO-CV is licensed with the [Mozilla Public License version
2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).

**Continuous integration status:**

[![Build
Status](https://travis-ci.org/DO-CV/DO-CV.svg?branch=develop)](https://travis-ci.org/DO-CV/DO-CV) <a href="https://ci.appveyor.com/project/davidok8/do-cv"><img src="https://ci.appveyor.com/api/projects/status/github/do-cv/do-cv?branch=develop&svg=true" alt="Build Status: Windows" /></a>
[![Coverage Status](https://coveralls.io/repos/DO-CV/DO-CV/badge.svg?branch=develop)](https://coveralls.io/r/DO-CV/DO-CV?branch=develop)

Documentation
-------------

Documentation is available [here](http://do-cv.readthedocs.org/en/latest/) at
the excellent [readthedocs.org](https://readthedocs.org/).

Also have a look at the legacy doxygen-based documentation
[here](http://do-cv.github.io/DO-CV/).


Why yet another library?
------------------------

I have written DO-CV during my PhD at the [IMAGINE
lab](http://imagine.enpc.fr/) in [Ecole des Ponts,
ParisTech](http://www.enpc.fr).

Historically, I started writing DO-CV before openCV came up with a new C++ API.
I used openCV for the first time during my research internship at Siemens. That
was in 2008 (quite some time, now that I think of it!). I was very frustrated
with it. After a while, I started writing the library as a hobby to have a more
easy-to-use library and also to gain a better mastery of the C++ language.

Today openCV has evolved a lot. Despite that, openCV has yet to convince me to
use it, API-wise. Another reason is that I also want to keep my library alive.


Build the libraries
-------------------

To build the libraries, run:

```
  sudo apt-get install cmake
  sudo apt-get install -qq qtbase5-dev

  mkdir build
  cd build
  cmake ..
  make  -j N  # N is the number of CPU cores you want to use.

  make test
```
