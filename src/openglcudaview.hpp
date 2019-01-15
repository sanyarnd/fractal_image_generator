#ifndef OPENGLCUDAVIEW_HPP
#define OPENGLCUDAVIEW_HPP

#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QOpenGLWidget>
#include <cuda_gl_interop.h>

//#include "image.hpp"

class OpenGLCudaView : public QOpenGLWidget, protected QOpenGLFunctions {
public:
  OpenGLCudaView(OpenGLCudaView const &) = delete;
  OpenGLCudaView(OpenGLCudaView &&) = delete;
  OpenGLCudaView &operator=(OpenGLCudaView const &) = delete;
  OpenGLCudaView &operator=(OpenGLCudaView &&) = delete;

  explicit OpenGLCudaView(QWidget *parent = nullptr);
  ~OpenGLCudaView() override;

  void initializeGL() override;
  void resizeGL(int w, int h) override;
  //  void paintGL() override;

public slots:
  void freeResources();

  // protected:
  //  void keyPressEvent(QKeyEvent *event);
  //  void keyReleaseEvent(QKeyEvent *event);
  //  void mousePressEvent(QMouseEvent *event);
  //  void mouseReleaseEvent(QMouseEvent *event);
  //  void wheelEvent(QWheelEvent *event);

private:
  QSize _size{};
  QOpenGLBuffer *_pix_buf;
  QOpenGLTexture *_texture;
  void *_cuda_buf{};
};

#endif // OPENGLCUDAVIEW_HPP
