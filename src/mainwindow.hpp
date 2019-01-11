#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <QComboBox>
#include <QGroupBox>
#include <QMainWindow>

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  Q_DISABLE_COPY(MainWindow)
  MainWindow(MainWindow &&) = delete;
  MainWindow &operator=(MainWindow &&) = delete;
  ~MainWindow() override = default;

private slots:
  // Underlying data is a simple pair of <QString, QSize>
  // by default ComboBox adds item text, but not item data
  // this small hack fixes it (it's also possible to
  // change this behaviour by subclassing QComboBox)
  void fixImageSizeQVariant(int index) const;

private:
  QComboBox *imageSize{};
};

#endif // MAINWINDOW_HPP
