#pragma once

#if defined __cplusplus
// Google Test
#include <gtest/gtest.h>
// DO-CV
#include <DO/Core.hpp>
// STL
#include <iostream>
#include <list>
#include <utility>
// Miscellaneous
# ifdef _WIN32
#   include <windows.h>
# else
#   include <unistd.h>
# endif
#endif