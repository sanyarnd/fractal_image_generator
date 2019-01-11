cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR) # LTS release
project(FlameFractalGenerator LANGUAGES CXX CUDA VERSION 1.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/3rdparty/GSL")


if(WIN32)
    set(Qt5Locator_FIND_QUIETLY ON)
    include(windows/Qt5Locator)
    set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build")
    set(GLUT_ROOT_PATH "${CMAKE_SOURCE_DIR}/3rdparty/windows/freeglut")
    set(GLUT_LIBRARY_DLL "${CMAKE_SOURCE_DIR}/3rdparty/windows/freeglut/bin/freeglut.dll")
    set(GLEW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/windows/glew/include")
    set(GLEW_LIBRARY "${CMAKE_SOURCE_DIR}/3rdparty/windows/glew/lib/glew32.lib")
    set(GLEW_LIBRARY_DLL "${CMAKE_SOURCE_DIR}/3rdparty/windows/glew/bin/glew32.dll")
    set(MS_GSL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/GSL/include")
endif()

find_package(OpenGL REQUIRED QUIET)
find_package(GLUT REQUIRED QUIET)
find_package(GLEW REQUIRED QUIET)
find_package(OpenCV 4.0.1 REQUIRED QUIET)
find_package(Qt5 5.12 COMPONENTS Widgets OpenGL REQUIRED QUIET)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_SOURCE_DIR}/ui")


add_executable(${PROJECT_NAME}
    src/main.cpp

    src/mainwindow.hpp
    src/mainwindow.cpp

    src/openglcudaview.hpp
    src/openglcudaview.cpp

    src/resolutionmodel.hpp
    src/resolutionmodel.cpp

    src/variation.hpp
    src/variation.cpp

    src/cuda_utils.cuh
    src/cuda_utils.cu
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"

    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${OpenCV_INCLUDE_DIRS}
        ${MS_GSL_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenGL::GL
        GLUT::GLUT
        GLEW::GLEW
        ${OpenCV_LIBS}
        Qt5::Widgets
        Qt5::OpenGL
)

############################
# Copy .dll's
############################
if(WIN32)
    get_target_property(__opencv_dbg opencv_world IMPORTED_LOCATION_DEBUG)
    get_target_property(__opencv_rel opencv_world IMPORTED_LOCATION_RELEASE)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying OpenCV .dll's to output directory"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:debug>:${__opencv_dbg}>$<$<CONFIG:release>:${__opencv_rel}>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

    # Freeglut and GLEW have no *Config.cmake
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying freeglut.dll to output directory"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${GLUT_LIBRARY_DLL}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Copying glew32.dll to output directory"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${GLEW_LIBRARY_DLL}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

    # Deploy Qt
    get_target_property(QT5_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(QT5_WINDEPLOYQT_EXECUTABLE ${QT5_QMAKE_EXECUTABLE} PATH)
    set(QT5_WINDEPLOYQT_EXECUTABLE "${QT5_WINDEPLOYQT_EXECUTABLE}/windeployqt.exe")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${QT5_WINDEPLOYQT_EXECUTABLE} $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif(WIN32)
