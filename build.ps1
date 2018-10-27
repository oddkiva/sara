$cmake_vsver = @{2015="v140"; 2017="v141"};
$conan_vsver = @{2015="14"; 2017="15"};

$vsver = 2015

$build_dir = "sara-build-vs$vsver"

$cmake_toolset = $cmake_vsver[$vsver]
$conan_compiler_version = $conan_vsver[$vsver]

iex "New-Item -ItemType directory -Path ..\build_dir"
iex "cd ..\$build_dir"
{
  iex "conan install ..\sara -s compiler.version=$toolset"

  $cmake_options  = "-DSARA_BUILD_VIDEOIO:BOOL=ON "
  $cmake_options += "-DSARA_BUILD_SHARED_LIBS:BOOL=ON "
  $cmake_options += "-DSARA_BUILD_SAMPLES:BOOL=ON "
  $cmake_options += "-DSARA_BUILD_TESTS:BOOL=ON"

  $cmake_command  = "cmake ..\sara -G'Visual Studio 15 2017 Win64' "
  $cmake_command += "-T $cmake_toolset "
  $cmake_command += "$cmake_options"
  iex "$cmake_command"
}
iex "cd ..\sara"
