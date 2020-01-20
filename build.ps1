$cmake_vsver = @{2015="v140"; 2017="v141"; 2019="v160"};
$cmake_vsver2 = @{2015="14"; 2017="15"; 2019="16"};
$build_shared_libs = @{"static"="OFF"; "shared"="ON"};

$vsver = 2017
$vsver2 = $cmake_vsver2[$vsver]

$build_type = "static"

$source_dir = "sara"
$build_dir = "sara-build-vs$vsver-$build_type"
$cmake_toolset = $cmake_vsver[$vsver]

echo "========================================================================="
echo "Checking if directory ..\$build_dir exists"
if (Test-Path ..\$build_dir) {
  echo "Removing existing directory to rebuild from scratch..."
  rm ..\$build_dir -r -fo
}

echo "Creating directory ..\$build_dir..."
iex "New-Item -ItemType directory -Path ..\$build_dir"
echo "`n"

# Configure the build solution with CMake.
echo "========================================================================="
echo "Configuring for CMake..."
$vcpkg_toolchain_file = "c:/vcpkg/scripts/buildsystems/vcpkg.cmake"

$cmake_options  = "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=$vcpkg_toolchain_file "
$cmake_options += "-DSARA_BUILD_VIDEOIO:BOOL=ON "
$cmake_options += "-DSARA_BUILD_SHARED_LIBS:BOOL=$($build_shared_libs[$build_type]) "
$cmake_options += "-DSARA_BUILD_SAMPLES:BOOL=ON "
$cmake_options += "-DSARA_BUILD_TESTS:BOOL=ON"

echo "CMake options = $cmake_options"
echo "`n"

# Invoke CMake command.
echo "========================================================================="
echo "Configuring for CMake..."
cd ..\$build_dir
$cmake_command  = "cmake -S `"..\$source_dir`" -B `".`" -G `"Visual Studio $vsver2 $vsver Win64`" "
# $cmake_command += "-T `"$cmake_toolset`" "
$cmake_command += "$cmake_options"

echo "$cmake_command"
iex "$cmake_command"
iex "cmake --build . --target ALL_BUILD"
return

