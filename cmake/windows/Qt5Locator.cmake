# based on: https://raw.githubusercontent.com/Artikash/Textractor/master/cmake/QtUtils.cmake

IF(MSVC)
    if(NOT Qt5Locator_FIND_QUIETLY)
        message(STATUS "Detecting Qt distro location")
    endif()

    # look for user-registry pointing to qtcreator
    get_filename_component(QT_BIN [HKEY_CURRENT_USER\\Software\\Classes\\Applications\\QtProject.QtCreator.pro\\shell\\Open\\Command] PATH)

    # get root path so we can search for 5.3, 5.4, 5.5, etc
    string(REPLACE "/Tools" ";" QT_BIN "${QT_BIN}")
    list(GET QT_BIN 0 QT_BIN)
    file(GLOB QT_VERSIONS "${QT_BIN}/5.*")
    list(SORT QT_VERSIONS)

    if(QT_VERSIONS)
        if(NOT Qt5Locator_FIND_QUIETLY)
            message(STATUS "-- Found Qt modules: '${QT_VERSIONS}'")
        endif()
    else()
        message(FATAL_ERROR "Cannot find any Qt 5.x distributions in Windows Registry (reconstructed path: '${QT_BIN}')")
    endif()

    # assume the latest version will be last alphabetically
    list(REVERSE QT_VERSIONS)

    list(GET QT_VERSIONS 0 QT_VERSION)
    if(NOT Qt5Locator_FIND_QUIETLY)
        message(STATUS "-- Using '${QT_VERSION}'")
    endif()

    # fix any double slashes which seem to be common
    string(REPLACE "//" "/"  QT_VERSION "${QT_VERSION}")

    if(MSVC_VERSION GREATER_EQUAL "1910")
        set(QT_MSVC "2017")
    elseif(MSVC_VERSION GREATER_EQUAL "1900")
        set(QT_MSVC "2015")
    else()
        # Latest QT versions >5.10 provides only 2015 and 2017 prebuilt binaries
        message(WARNING "Unsupported MSVC toolchain version")
    endif()

    if(QT_MSVC)
        # check for 64-bit target
        if(CMAKE_CL_64)
            SET(QT_MSVC "${QT_MSVC}_64")
        endif()

        set(QT_TOLLCHAIN "${QT_VERSION}/msvc${QT_MSVC}")
        if(EXISTS ${QT_TOLLCHAIN})
            set(Qt5_DIR "${QT_TOLLCHAIN}/lib/cmake/Qt5")
        elseif(QT_MSVC EQUAL "2017")
            #2017 is ABI compatible with 2015
            if(CMAKE_CL_64)
                set(QT_TOLLCHAIN "${QT_VERSION}/msvc2015_64")
            else()
                set(QT_TOLLCHAIN "${QT_VERSION}/msvc2015")
            endif()

            if(EXISTS ${QT_TOLLCHAIN})
                set(Qt5_DIR "${QT_TOLLCHAIN}/lib/cmake/Qt5")
            else()
                message(FATAL_ERROR "Required QT5 toolchain is not installed")
            endif()
        else()
            message(FATAL_ERROR "Required QT5 toolchain is not installed")
        endif()
    endif()
ENDIF()
