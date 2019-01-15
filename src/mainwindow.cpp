#include "mainwindow.hpp"

#include <QDebug>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QRegExp>
#include <QRegExpValidator>
#include <QSizePolicy>
#include <QSplitter>
#include <numeric>

#include "cuda_utils.cuh"
#include "openglcudaview.hpp"
#include "resolutionmodel.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow{parent} {
  createUi();
  setupUi();
}

void MainWindow::fixImageSizeQVariant(int index) const {
  auto data = imageSize->itemData(index);
  if (!data.isValid()) {
    auto text = imageSize->itemText(index);
    auto s = text.split(QRegExp(R"(\s*x\s*)"));
    imageSize->setItemData(index, QSize{s[0].toInt(), s[1].toInt()});
  }
}

void MainWindow::createUi() {
  // create scrollview for GL widget
  glView = new OpenGLCudaView();
  glView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  // populate control box
  auto controlGroup = new QGroupBox();
  controlGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  auto controlGroupLayout = new QFormLayout();
  controlGroup->setTitle(tr("Controls"));
  controlGroup->setLayout(controlGroupLayout);

  start = new QPushButton(tr("Start"));
  stop = new QPushButton(tr("Stop"));
  auto launchLayout = new QHBoxLayout();
  launchLayout->addWidget(start);
  launchLayout->addWidget(stop);
  controlGroupLayout->addRow(launchLayout);

  auto cudaDeviceName = new QLabel();
  auto props = cuda_info::devicePropertries();
  cudaDeviceName->setText(static_cast<char const *>(props.name));
  controlGroupLayout->addRow(tr("Device"), cudaDeviceName);

  imageSize = new QComboBox();
  imageSize->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Image Size"), imageSize);

  seed = new QSpinBox();
  seed->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Seed"), seed);

  samples = new QSpinBox();
  samples->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Samples"), samples);

  iterations = new QSpinBox();
  iterations->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Iterations"), iterations);

  symmetry = new QSpinBox();
  symmetry->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Rotation"), symmetry);

  supersampling = new QSpinBox();
  supersampling->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
  controlGroupLayout->addRow(tr("&Supersampling"), supersampling);

  addXform = new QPushButton(tr("Add"));
  removeXform = new QPushButton(tr("Remove"));
  auto xformLayout = new QHBoxLayout();
  xformLayout->addWidget(addXform);
  xformLayout->addWidget(removeXform);
  xformLayout->setStretchFactor(addXform, 1);
  xformLayout->setStretchFactor(removeXform, 1);
  controlGroupLayout->addRow(tr("XForm"), xformLayout);

  xformList = new QTreeView();
  xformList->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  auto sp = xformList->sizePolicy();
  sp.setVerticalStretch(1);
  xformList->setSizePolicy(sp);
  controlGroupLayout->addRow(xformList);

  loadXforms = new QPushButton(tr("Load..."));
  saveXforms = new QPushButton(tr("Save..."));
  auto fileOpLayout = new QHBoxLayout();
  fileOpLayout->addWidget(loadXforms);
  fileOpLayout->addWidget(saveXforms);
  controlGroupLayout->addRow(fileOpLayout);

  // set central widget
  auto splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(glView);
  splitter->addWidget(controlGroup);
  splitter->setStretchFactor(0, 1);
  splitter->setStretchFactor(1, 0);
  splitter->setCollapsible(0, false);
  splitter->setCollapsible(1, false);

  setCentralWidget(splitter);
}

void MainWindow::setupUi() {
  setWindowTitle(tr("Flame fractal generator"));
  setMinimumSize({800, 600});

  imageSize->addItem("800x600", QSize(800, 600));
  imageSize->addItem("1280x720", QSize(1280, 720));
  imageSize->addItem("1600x900", QSize(1600, 900));
  imageSize->addItem("1920x1080", QSize(1920, 1080));
  imageSize->addItem("3840x2160", QSize(3840, 2160));

  imageSize->setEditable(true);
  imageSize->setInsertPolicy(QComboBox::InsertAtTop);
  imageSize->setValidator(new QRegExpValidator(
      QRegExp(R"([1-9]\d{0,5}\s*x\s*[1-9]\d{0,5})"), this));
  connect(imageSize, SIGNAL(activated(int)), this,
          SLOT(fixImageSizeQVariant(int)));

  seed->setRange(std::numeric_limits<int>::min() + 1,
                 std::numeric_limits<int>::max() - 1);
  seed->setValue(42);
  seed->setSingleStep(1000);
  seed->setToolTip(tr("Random seed value"));

  samples->setRange(1, std::numeric_limits<int>::max() - 1);
  samples->setValue(30000);
  samples->setSingleStep(1000);
  samples->setToolTip(tr("Number of passes for each variation [1, +oo]"));

  iterations->setRange(1, std::numeric_limits<int>::max() - 1);
  iterations->setValue(1000);
  iterations->setSingleStep(100);
  iterations->setToolTip(tr("Number of iterations per sample [1, +oo]"));

  symmetry->setRange(1, 72);
  symmetry->setValue(1);
  symmetry->setToolTip(tr("Number of rotations [1, 72]"));

  supersampling->setRange(1, 16);
  supersampling->setValue(1);
  supersampling->setToolTip(tr("Render image in bigger resolution [1, 16]"));
}
