$cmake_vsver = @{2015="v140"; 2017="v141"; 2019="v160"};
$cmake_vsver2 = @{2015="14"; 2017="15"; 2019="16"};
$build_shared_libs = @{"static"="OFF"; "shared"="ON"};

$vsver = 2017
$vsver2 = $cmake_vsver2[$vsver]

$build_type = "shared"

$source_dir = $pwd
$build_dir = "sara-build-vs$vsver-$build_type"
$cmake_toolset = $cmake_vsver[$vsver]

$boost_dir = "C:\local\boost_1_71_0"
$halide_dir = "C:\local\halide"
$cudnn_dir = "C:\local\cudnn"
$tensorrt_dir = "C:\local\TensorRT-7.0.0.11"
$nvidia_codec_sdk_dir = "C:\local\Video_Codec_SDK_9.1.23"

$update_vcpkg = $false



if ($update_vcpkg) {
  echo "========================================================================="
  echo "Install dependencies with vcpkg..."
  cd c:/vcpkg/
  git pull
  .\bootstrap-vcpkg.bat

  # Install Image I/O libraries.
  iex ".\vcpkg.exe install libjpeg-turbo:x64-windows"
  iex ".\vcpkg.exe install libpng:x64-windows"
  iex ".\vcpkg.exe install tiff:x64-windows"

  # Install Video I/O libraries.
  iex ".\vcpkg.exe install ffmpeg:x64-windows"

  # Install HDF5 libraries.
  iex ".\vcpkg.exe install hdf5[cpp]:x64-windows"

  # Install Ceres libraries.
  iex ".\vcpkg.exe install ceres[cxsparse,suitesparse]:x64-windows"

  # Install GLEW libraries.
  iex ".\vcpkg.exe install glew:x64-windows"
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
echo "Checking if directory ..\$build_dir exists"
if (Test-Path ..\$build_dir) {
  echo "Removing existing directory to rebuild from scratch..."
  rm ..\$build_dir -r -fo
}

echo "Creating directory `"..\$build_dir`" ..."
iex "New-Item -ItemType directory -Path ..\$build_dir"
echo "`n"



echo "========================================================================="
echo "Configuring for CMake..."
$vcpkg_toolchain_file = "c:/vcpkg/scripts/buildsystems/vcpkg.cmake"

$cmake_options  = "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=$vcpkg_toolchain_file "
$cmake_options += "-DBOOST_ROOT:PATH=$boost_dir "
$cmake_options += "-DHALIDE_DISTRIB_DIR:PATH=$halide_dir "
$cmake_options += "-DCMAKE_PREFIX_PATH=`"$cudnn_dir;$tensorrt_dir`" "
$cmake_options += "-DSARA_BUILD_VIDEOIO:BOOL=ON "
$cmake_options += "-DSARA_BUILD_SHARED_LIBS:BOOL=$($build_shared_libs[$build_type]) "
$cmake_options += "-DSARA_BUILD_SAMPLES:BOOL=ON "
$cmake_options += "-DSARA_BUILD_TESTS:BOOL=ON "
$cmake_options += "-DSARA_USE_HALIDE:BOOL=ON "
$cmake_options += "-DNvidiaVideoCodec_ROOT=$nvidia_codec_sdk_dir "

echo "CMake options = $cmake_options"
echo "`n"

cd ..\$build_dir
$cmake_command  = "cmake -S `"..\sara`" -B `".`" -G `"Visual Studio $vsver2 $vsver Win64`" "
$cmake_command += "-T `"$cmake_toolset`" "
$cmake_command += "$cmake_options"
iex "$cmake_command"
echo "`n"



echo "========================================================================="
echo "Building the libraries in Debug mode..."
iex "cmake --build . --target ALL_BUILD --config Debug -j12"
iex "ctest --output-on-failure -C `"Debug`" -j12"

echo "Building the libraries in Release mode..."
iex "cmake --build . --target ALL_BUILD --config Release -j12"
iex "ctest --output-on-failure -C `"Release`" -j12"



Write-Host -NoNewLine 'Press any key to exit...';
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown');
