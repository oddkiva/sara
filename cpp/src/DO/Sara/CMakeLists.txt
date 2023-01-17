list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include(UseDOSaraCore)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  return()
endif ()

include(UseDOSaraFileSystem)
include(UseDOSaraGraphics)
include(UseDOSaraImageIO)
if (SARA_BUILD_VIDEOIO)
  include(UseDOSaraVideoIO)
endif ()

include(UseDOSaraDisjointSets)

include(UseDOSaraKDTree)
include(UseDOSaraImageProcessing)
include(UseDOSaraGeometry)

include(UseDOSaraFeatures)
include(UseDOSaraFeatureDetectors)
include(UseDOSaraFeatureDescriptors)

include(UseDOSaraMatch)
include(UseDOSaraFeatureMatching)
include(UseDOSaraMultiViewGeometry)
include(UseDOSaraRANSAC)