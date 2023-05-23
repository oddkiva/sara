$cmake_vsver = @{2015="v140"; 2017="v141"; 2019="v1421"; 2022="v143"};
$cmake_vsver2 = @{2015="14"; 2017="15"; 2019="16"; 2022="17"};
$build_shared_libs = @{"static"="OFF"; "shared"="ON"};

$vsver = 2022
$vsver2 = $cmake_vsver2[$vsver]

$build_type = "static"

$source_dir = $pwd
$build_dir = "sara-build-vs$vsver-$build_type"
$cmake_toolset = $cmake_vsver[$vsver]

$qt_dir = "C:\local\qt-everywhere-src-6.1.2\qtbase"
$cudnn_dir = "C:\local\C:\local\cudnn-windows-x86_64-8.8.0.121_cuda12-archive"
$halide_dir = "C:\local\Halide-15.0.0-x86-64-windows"
$nvidia_codec_sdk_dir = "C:\local\Video_Codec_SDK_9.1.23"
$tensorrt_dir = "C:\local\TensorRT-8.6.0.12.Windows10.x86_64.cuda-12.0"

$update_vcpkg = $false
$build_from_scratch = $true
$run_cmake_build = $false


if ($update_vcpkg) {
  echo "========================================================================="
  echo "Install dependencies with vcpkg..."
  cd c:/vcpkg/
  git pull
  .\bootstrap-vcpkg.bat

  iex ".\vcpkg.exe update"
  iex ".\vcpkg.exe upgrade"

  # Install Boost libraries.
  iex ".\vcpkg.exe install boost:x64-windows"

  # Install Image I/O libraries.
  iex ".\vcpkg.exe install libjpeg-turbo:x64-windows"
  iex ".\vcpkg.exe install libpng:x64-windows"
  iex ".\vcpkg.exe install tiff:x64-windows"
  iex ".\vcpkg.exe install libheif:x64-windows"
  iex ".\vcpkg.exe install libwebp:x64-windows"

  # Install Video I/O libraries.
  iex ".\vcpkg.exe install ffmpeg:x64-windows"

  # Install HDF5 libraries.
  iex ".\vcpkg.exe install hdf5[cpp]:x64-windows"

  # Install Ceres libraries.
  iex ".\vcpkg.exe install ceres[cxsparse,suitesparse]:x64-windows"

  # Install Halide libraries.
  #
  # Please don't... It builds too many things and takes too much space:
  # - LLVM
  # - etc.
  # iex ".\vcpkg.exe install halide:x64-windows"

  # Install GLEW libraries.
  iex ".\vcpkg.exe install glew:x64-windows"

  # Install GLFW libraries.
  iex ".\vcpkg.exe install glfw3:x64-windows"
  echo `n
}

# Go back to the source directory
cd $source_dir



echo "========================================================================="
echo "Installing Python packages"
iex "pip install -U pip"
iex "pip install -r requirements.txt"
echo `n



echo "========================================================================="
if ($build_from_scratch) {
  echo "Checking if directory ..\$build_dir exists"
  if (Test-Path ..\$build_dir) {
    echo "Removing existing directory to rebuild from scratch..."
    rm ..\$build_dir -r -fo
  }

  echo "Creating directory `"..\$build_dir`" ..."
  iex "New-Item -ItemType directory -Path ..\$build_dir"
  echo "`n"
}


echo "========================================================================="
echo "Configuring for CMake..."
$vcpkg_toolchain_file = "c:/vcpkg/scripts/buildsystems/vcpkg.cmake"

$cmake_options  = "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=$vcpkg_toolchain_file "
$cmake_options += "-DCMAKE_PREFIX_PATH=`"$cudnn_dir;$tensorrt_dir;$qt_dir`" "
$cmake_options += "-DHALIDE_DISTRIB_DIR:PATH=$halide_dir "
$cmake_options += "-DHalideHelpers_DIR:PATH=$halide_dir\lib\cmake\HalideHelpers "
$cmake_options += "-DSARA_USE_QT6:BOOL=ON "
$cmake_options += "-DSARA_BUILD_VIDEOIO:BOOL=ON "
$cmake_options += "-DSARA_BUILD_SHARED_LIBS:BOOL=$($build_shared_libs[$build_type]) "
$cmake_options += "-DSARA_BUILD_SAMPLES:BOOL=ON "
$cmake_options += "-DSARA_BUILD_TESTS:BOOL=ON "
$cmake_options += "-DSARA_BUILD_DRAFTS:BOOL=ON "
$cmake_options += "-DSARA_USE_HALIDE:BOOL=ON "
$cmake_options += "-DNvidiaVideoCodec_ROOT=$nvidia_codec_sdk_dir "

echo "CMake options = $cmake_options"
echo "`n"

cd ..\$build_dir
$cmake_command  = "cmake -S `"..\sara`" -B `".`" -G `"Visual Studio $vsver2 $vsver`" "
# Not applicable to VS 2019.
# $cmake_command += "-T `"$cmake_toolset`" "
$cmake_command += "$cmake_options"
iex "$cmake_command"
echo "`n"



if ($run_cmake_build) {
  echo "========================================================================="
  echo "Building the libraries in Debug mode..."
  iex "cmake --build . --target ALL_BUILD --config Debug -j12"
  iex "ctest --output-on-failure -C `"Debug`" -j12"

  echo "Building the libraries in Release mode..."
  iex "cmake --build . --target ALL_BUILD --config Release -j12"
  iex "ctest --output-on-failure -C `"Release`" -j12"
}



Write-Host -NoNewLine 'Press any key to exit...';
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');
