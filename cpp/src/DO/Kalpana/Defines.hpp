// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#if defined(_WIN32) || defined(_WIN32_WCE)
# ifdef DO_KALPANA_EXPORTS
#   define DO_KALPANA_EXPORT __declspec(dllexport) /* We are building the libraries */
# elif defined(DO_KALPANA_STATIC)
#   define DO_KALPANA_EXPORT
# else
#   define DO_KALPANA_EXPORT __declspec(dllimport) /* We are using the libraries */
# endif
#else
# define DO_KALPANA_EXPORT
#endif

#ifdef DO_KALPANA_DEPRECATED
# undef DO_KALPANA_DEPRECATED
#endif

#ifdef __GNUC__
# define DO_KALPANA_DEPRECATED __attribute__ ((deprecated))
#elif defined(_MSC_VER)
# define DO_KALPANA_DEPRECATED __declspec(deprecated)
#else
# pragma message("WARNING: You need to implement DO_KALPANA_DEPRECATED for this compiler")
# define DO_KALPANA_DEPRECATED
#endif

//! @}
