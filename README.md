Sara: C++ Computer Vision Library
=================================

**I HAVE MOVED TO GITLAB. PLEASE GOTO: https://gitlab.com/DO-CV/sara**

**SO LONG GITHUB!**

[![GitLab CI Build Status](https://gitlab.com/DO-CV/sara/badges/master/pipeline.svg)](https://gitlab.com/DO-CV/sara/-/pipelines)
[![GitHub Actions Build Status](https://github.com/DO-CV/sara/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/DO-CV/sara/actions)
<a href="https://ci.appveyor.com/project/davidok8/do-cv"><img src="https://ci.appveyor.com/api/projects/status/github/do-cv/sara?branch=master&svg=true" alt="Build Status: Windows" /></a>
[![Travis Build Status](https://travis-ci.org/DO-CV/sara.svg?branch=master)](https://travis-ci.org/DO-CV/sara)
[![Coverage Status](https://coveralls.io/repos/DO-CV/sara/badge.svg?branch=master)](https://coveralls.io/r/DO-CV/sara?branch=master)
[![License](https://img.shields.io/badge/license-MPL2-blue.svg)](LICENSE)
[![Codedocs Reference API Documentation](https://codedocs.xyz/DO-CV/sara.svg)](https://codedocs.xyz/DO-CV/sara/)

*Sara* (सार) is a Sanskrit word meaning *essence*.

*Sara* tries to focus on:

1. having an **easy-to-use and simple API**;
2. having **easy-to-understand ~~and efficient~~** implementations of computer vision
   algorithms;
   - Well "efficient" if you try to use my Halide or CUDA backends
     for example. "Easy-to-read" thanks to the exciting advances in compiler
     technologies like Halide, so we don't need to manipulate CPU
     intrinsics directly on the mainstream architectures.
     I have documented that some of my image processing implementations perform
     much better than OpenCV's CPU/GPU implementations, much to my surprise...
     See [here](https://gitlab.com/DO-CV/sara/-/blob/master/python/do/sara/benchmark/image_processing.py).

   - Also *Sara* can also decode videos by using nVidia's hardware
     acceleration. It is indeed very fast on 4K videos (3840x2160).
3. **~~rigorous~~ sufficiently good testing**
   - Well, as much as my limited time will permit it.

As a side note, as of 2021, I realize that yes... the ignorance of youth made me
inflate my statements without realizing it. This project has always been
a one-man project. Now I have more experience, so this has become a bit truer.

The design of *Sara* libraries is driven by the KISS principle. I enjoy
maintaining *Sara* and making it evolve when I feel like it.

*Sara* is licensed with the [Mozilla Public License version
2.0](https://github.com/DO-CV/DO-CV/raw/master/COPYING.MPL2).

**Since June 12, 2019, Sara is now a C++ 17 only project.** The improvements in
the language are compelling enough to make the library more readable.

**Supported compilers:**
- Visual Studio 2017 and above
- GCC 7 and above


Documentation
-------------

I don't have much time to maintain the documentation. **I'd love your help**.

What I can easily do is to keep up-to-date the [reference API
documentation](https://codedocs.xyz/DO-CV/sara/) hosted in **codedocs.xyz**.

~~I also write [documentation](https://sara-github.readthedocs.org/) at
**readthedocs.org** where I provide more mathematical details about my
implementations.~~ Still available but a bit outdated.

In any case, you are always better off consulting the [examples
folder](https://gitlab.com/DO-CV/sara/tree/master/cpp/examples) and the [test
folder](https://gitlab.com/DO-CV/sara/tree/master/cpp/test).

The codes are usually short, so they should help you to get up to speed with the
library usage.


Why yet another library?
------------------------

Of course, you should not use my library and use *OpenCV* instead. But keep on
reading what I have to say.

I like crafting software from the ground up. In particular, I like understanding
computer vision algorithms in-depth by reimplementing them from A to Z. Besides,
not everybody likes *OpenCV* and uses it.

I used OpenCV's C API for the first time when I did my research internship at
Siemens in 2008. And I was not aware of its C++ API at all. It was a very
frustrating experience, especially for the inexperienced programmer that I was
back then.

After a while, I started writing the library as a hobby to have a more
easy-to-use library and also to gain a better mastery of the C++ language. Then
the library *Sara* became more structured in 2009, when I started my PhD at the
[IMAGINE lab](http://imagine.enpc.fr/) in [Ecole des Ponts,
ParisTech](http://www.enpc.fr).

You can reuse parts of the library in your applications in the industry as I do
myself. The library was initially not designed for real-time critical
applications. Some algorithms can scale well. Some do not. In most use cases,
the library should be fine.

Regarding real-time processing, the way we achieve it will depend on the
platform. In non-mainstream platforms or low-power devices, *OpenCV* will not be
able to rescue you. However, I do want to point out that *Sara* can offer the
following:

- On CUDA platforms, some of my CUDA implementations are a good start and even
  better than OpenCV's.
- On low-power devices, the Halide CPU backend provides a good start. Then we
  would then need to optimize the scheduling code for non-desktop architectures.

Time has flown. Years after years, I am still keeping the library alive but
things happen a lot more slowly now.


Build the libraries
-------------------

You can have a look at the CI scripts like `.gitlab-ci.yml` or `.travis.yml`**
to have a minimal build.

To help you get started, you can try the Docker container. Assuming that you
have a CUDA GPU device and nvidia-docker installed, then:

1. Grab the Docker image at `registry.gitlab.com/do-cv/sara`
2. Run the docker container as I do:
   ```
   docker run --gpus all -it \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v "$HOME/.Xauthority:/root/.Xauthority:rw" \
       -v /media/Linux\ Data:/media/Linux\ Data \
       -v $PWD:/workspace/sara \
       -e DISPLAY \
       --ipc=host \
       --net=host \
       ${SARA_DOCKER_IMAGE} \
       /bin/zsh
   ```
3. Try running the build script: `./build.sh Release`.
4. Pray that it will work.

If it does not work, help me and try fixing it? =P
