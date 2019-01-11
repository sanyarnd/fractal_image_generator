# Flame fractal image generator
This program implements the following algorithm (see [[1]](paper/flame_draves.pdf) or [[2]](https://flam3.com/flame_draves.pdf)) with CUDA acceleration.

# Screenshots

# How to build
## Compilers and libraries
The program is tested to compile with the following toolchains:
* `MSVC 19.16.27025.1`
* `GCC 8.2`

Program uses the following third party libraries and utilities:
* `CMake 3.10.2` or higher **[external]**
* `Qt 5.12` **[external]**
* `CUDA 10.0.130.411.31` **[external]**
* `OpenCV 4.0.1` **[external]**
* `FreeGLUT 3.0.0` [bundled]
* `GLEW 2.1.0` [bundled]
* `LodePNG 20181230` [bundled]
* `GSL 2.0.0` [bundled]

## Compilation
You'll need to aquire and install all the things marked as `[external]`. It is very straightforward on `Linux` (install everything from your distro repositories), so I'll explain only the `Windows-specific` things.

### Windows
**Qt**: don't use archive bundle, use installer, otherwise you'll have to alternate CMake helper module (see [`windows/Qt5Locator`](cmake/windows/Qt5Locator.cmake)).

**OpenCV**: download pre-build package and unzip it to `3rdparty/opencv` (you must have the following items `3rdparty/opencv/{build, sources, ...}`). Project uses static linkage, so you don't need to copy any `.dll`'s.

#### Notes
I am using official pre-built **FreeGLUT** and **GLEW** packages with some file tree modifications to make it fit for `FintGLUT`/`FindGLEW` modules. There are too many steps, and so I decided to include them in repository instead (extra ~3.5mb).
