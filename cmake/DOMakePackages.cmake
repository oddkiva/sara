# NSIS specific options

# Installers for 32- vs. 64-bit CMake:
#  - Root install directory (displayed to end user at installer-run time)
#  - "NSIS package/display name" (text used in the installer GUI)
#  - Registry key used to store info about the installation
if(CMAKE_CL_64)
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} (Win64)")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} (Win64)")
else()
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
    set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
    set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
        "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
endif()


set(CPACK_NSIS_MODIFY_PATH ON)
# Set environment variables at installation
set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    WriteRegExpandStr HKCU \\\"Environment\\\" DOPLUSPLUS \\\"$INSTDIR\\\"
    SendMessage \\\${HWND_BROADCAST} \\\${WM_WININICHANGE} 0 \\\"STR:Environment\\\" /TIMEOUT=5000
    ")
# Unset environment variables at uninstallation
set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
    DeleteRegValue HKCU \\\"Environment\\\" DOPLUSPLUS
    SendMessage \\\${HWND_BROADCAST} \\\${WM_WININICHANGE} 0 \\\"STR:Environment\\\" /TIMEOUT=5000
    ")

# Go !!!
include(CPack)
