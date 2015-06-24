# USAGE: find_package(EasyEXIF)
#
#  This macro tries to locate the EasyEXIF library.
#
#  Uses the following variables to hint at paths:
#  - EasyEXIF_ROOT_DIR: EasyEXIF install dir.
#
#  The following variables are set:
#  - EasyEXIF_INCLUDE_DIR: the absolute path of the EasyEXIF include directory.
#  - EasyEXIF_LIBRARIES: the list of EasyEXIF static libraries.

set(PATH_HINTS
    ${EasyEXIF_ROOT_DIR}
    "/usr"
    "/usr/local"
    "C:/Program Files/DO-Sara")

find_path(
	EasyEXIF_INCLUDE_DIR exif.h
	PATHS ${PATH_HINTS}
	PATH_SUFFIXES include/easyexif
)
if(NOT EasyEXIF_INCLUDE_DIR)
  message(FATAL_ERROR
    "EasyEXIF include directory not found. "
		"Set EasyEXIF_ROOT_DIR for the EasyEXIF install dir."
	)
endif()
include_directories("${EasyEXIF_INCLUDE_DIR}")

find_library(
	EasyEXIF_LIBRARY
  NAMES easyexif
	PATHS ${PATH_HINTS}
	PATH_SUFFIXES lib
)

set(EasyEXIF_LIBRARIES ${EasyEXIF_LIBRARY})
