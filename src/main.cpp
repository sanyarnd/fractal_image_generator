#include <GL/freeglut.h>

#include <QApplication>
#include <QDateTime>
#include <QFile>
#include <QMessageBox>
#include <QMutex>
#include <QMutexLocker>
#include <QTextStream>

#include "cuda_utils.cuh"
#include "mainwindow.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);

  // cuda check
  if (!cuda_info::deviceIsPresent()) {
    QMessageBox::critical(
        nullptr, "No CUDA-capable devices",
        "The program cannot find any CUDA-capable devices.\n"
        "Please be sure that GPU device is active, especially if you have "
        "NVIDIA Optimus compatible GPU.");
    exit(-1);
  }

  MainWindow w;
  w.show();

  return QApplication::exec();
}

// int main(int argc, char *argv[]) {
//  auto data = new pixel[6];
//  gsl::multi_span<pixel, 2, 3> ms =
//      gsl::as_multi_span(data, gsl::dim(3), gsl::dim(2));
//  std::cout << ms[{1, 0}].density;
//  //  auto span = gsl::multi_span(data, gsl::dim(2), gsl::dim(3));
//  //  auto x = span.bounds();
//  //  std::cout << x.extent(0) << "\n";
//  //  span[0][1].
//  //  Image img(span);

//  //  QApplication a(argc, argv);
//  //  MainWindow w;

//  //  w.show();

//  //  return a.exec();
//}
