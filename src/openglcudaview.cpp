#include "openglcudaview.hpp"

#include <QDebug>

#include "cuda_utils.cuh"

OpenGLCudaView::OpenGLCudaView(QWidget *parent)
    : QOpenGLWidget{parent}, QOpenGLFunctions{} {
  QSurfaceFormat format;
  format.setVersion(3, 0);
  format.setSamples(4);
  format.setProfile(QSurfaceFormat::NoProfile);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
  setFormat(format);
}

OpenGLCudaView::~OpenGLCudaView() {}

void OpenGLCudaView::initializeGL() { initializeOpenGLFunctions(); }

void OpenGLCudaView::resizeGL(int w, int h) { qDebug() << "Resize"; }

void OpenGLCudaView::freeResources() {}
