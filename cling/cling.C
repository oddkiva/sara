// Eigen
#pragma cling add_include_path("./cpp/third-party/eigen")
#pragma cling add_include_path("./cpp/third-party/flann/src/cpp")
//#pragma cling load("../sara-build-Debug/lib/libflann_cpp_s-d.a")

// Qt 5
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtCore")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtConcurrent")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtGui")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtNetwork")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtQuick")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtQml")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtSql")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtWidgets")
#pragma cling add_include_path("/usr/include/x86_64-linux-gnu/qt5/QtXml")
#pragma cling add_include_path("/usr/lib/x86_64-linux-gnu/qt5/mkspecs/linux-g++-64")
#pragma add_library_path("/usr/lib/x86_64-linux-gnu")
#pragma cling load("/usr/lib/x86_64-linux-gnu/libQt5Core.so")
#pragma cling load("/usr/lib/x86_64-linux-gnu/libQt5Widgets.so")
#pragma cling load("/usr/lib/x86_64-linux-gnu/libQt5OpenGL.so")


#pragma cling add_include_path("/usr/local/include")
#pragma add_library_path("/usr/local/lib")
#pragma cling load("/usr/local/lib/libboost_filesystem.so")

// Sara
#pragma cling add_include_path("./cpp/src")
#pragma cling add_include_path("../sara-build-Debug/cpp/src")
#pragma cling add_library_path("../sara-build-Debug/lib")
#pragma cling load("../sara-build-Debug/lib/libDO_Sara_Core-d.so")
#pragma cling load("../sara-build-Debug/lib/libDO_Sara_FileSystem-d.so")
#pragma cling load("../sara-build-Debug/lib/libDO_Sara_Graphics-d.so")
