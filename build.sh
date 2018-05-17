#!/bin/bash
set -ex

function install_python_packages_via_pip()
{
  pip install numpy nose
}

function build_library()
{
  local cmake_options="-DCMAKE_BUILD_TYPE=${sara_build_type} "
  cmake_options+="-DCMAKE_EXPORT_COMPILE_COMMANDS=1 "
  cmake_options+="-DCMAKE_PREFIX_PATH=/usr/local/Qt-5.10.0 "
  cmake_options+="-DSARA_BUILD_VIDEOIO=ON "
  cmake_options+="-DSARA_BUILD_PYTHON_BINDINGS=ON "
  cmake_options+="-DSARA_BUILD_SHARED_LIBS=ON "
  cmake_options+="-DSARA_BUILD_TESTS=ON "
  cmake_options+="-DSARA_BUILD_SAMPLES=ON "

  # Generate makefile project.
  cmake ../sara ${cmake_options}

  # Build the library.
  make -j`nproc` && make test && make pytest && make package
}

function install_package()
{
  if [ -f "/etc/debian_version" ]; then
    # Register the package to the local debian repository.
    dpkg-sig --sign builder libDO-Sara-shared-*.deb
    sudo cp libDO-Sara-shared-*.deb /usr/local/debs
    sudo update-local-debs
    sudo apt-get update
    sudo apt-get install --reinstall libdo-sara-shared
  else
    rpm_package_name=$(echo `ls *.rpm`)
    sudo rpm -ivh --force ${rpm_package_name}
  fi
}


if [ -z ${1+x} ]; then
  sara_build_type="Release"
else
  sara_build_type=$1
fi
echo "sara_build_type=${sara_build_type}"

if [ -z ${2+x} ]; then
  sara_build_dir="sara-build"
else
  sara_build_dir=$2
fi
echo "sara_build_dir=${sara_build_dir}"

# Create the build directory.
if [ -d "../${sara_build_dir}" ]; then
  rm -rf ../${sara_build_dir}
fi

mkdir ../${sara_build_dir}


cd ../${sara_build_dir}
{
  install_python_packages_via_pip
  build_library
  #install_package
}
cd ..
