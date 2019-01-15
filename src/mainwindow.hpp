#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QComboBox>
#include <QMainWindow>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QTextEdit>
#include <QTreeView>

#include "openglcudaview.hpp"

class MainWindow : public QMainWindow {
Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  Q_DISABLE_COPY(MainWindow)
  MainWindow(MainWindow &&) = delete;
  MainWindow &operator=(MainWindow &&) = delete;
  ~MainWindow() override = default;

private slots:
  // Underlying data is a simple pair of <QString, QSize>;
  // by default ComboBox adds item text, and no item data;
  // this small hack fixes it (it's also possible to
  // change this behaviour by subclassing QComboBox)
  void fixImageSizeQVariant(int index) const;

private:
  OpenGLCudaView *glView{};
  QPushButton *start{};
  QPushButton *stop{};
  QComboBox *imageSize{};
  QSpinBox *seed;
  QSpinBox *samples{};
  QSpinBox *iterations{};
  QSpinBox *symmetry{};
  QSpinBox *supersampling{};

  QPushButton *addXform{};
  QPushButton *removeXform{};

  QTreeView *xformList{};
  QPushButton *loadXforms{};
  QPushButton *saveXforms{};

  // create GUI components (ui related stuff, like layout, names, tooltips etc)
  void createUi();
  // initialize GUI components (signals/slots, constraints, any non-ui related
  // stuff)
  void setupUi();
};

#endif // MAINWINDOW_HPP
