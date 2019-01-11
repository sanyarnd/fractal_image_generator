#include "src/mainwindow.hpp"

#include <QComboBox>
#include <QDebug>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegExp>
#include <QRegExpValidator>
#include <QScrollArea>
#include <QSizePolicy>
#include <QSpinBox>
#include <QSplitter>
#include <QTreeView>

#include <numeric>

#include "cuda_utils.cuh"
#include "openglcudaview.hpp"
#include "resolutionmodel.hpp"

MainWindow::MainWindow(QWidget *parent) : QMainWindow{parent} {
  setWindowTitle(tr("Flame fractal generator"));
  setMinimumSize({640, 480});

  // create scrollview for GL widget
  auto viewScroll = new QScrollArea();
  viewScroll->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  //  auto view = new OpenGLCudaView();
  //  view->resize({640, 640});
  //  viewScroll->setWidget(view);

  // populate control box
  auto controlGroup = new QGroupBox();
  controlGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
  auto controlGroupLayout = new QFormLayout();
  controlGroup->setTitle(tr("Controls"));
  controlGroup->setLayout(controlGroupLayout);

  auto start = new QPushButton(tr("Start"));
  auto stop = new QPushButton(tr("Stop"));
  auto launchLayout = new QHBoxLayout();
  launchLayout->addWidget(start);
  launchLayout->addWidget(stop);
  controlGroupLayout->addRow(launchLayout);

  auto cudaDeviceName = new QLabel();
  auto props = cudaDevicePropertries();
  cudaDeviceName->setText(static_cast<char const *>(props.name));
  controlGroupLayout->addRow(tr("Device"), cudaDeviceName);

  imageSize = new QComboBox();
  imageSize->setEditable(true);
  imageSize->setInsertPolicy(QComboBox::InsertAtTop);
  imageSize->addItem("800x600", QSize(800, 600));
  imageSize->addItem("1280x720", QSize(1280, 720));
  imageSize->addItem("1600x900", QSize(1600, 900));
  imageSize->addItem("1920x1080", QSize(1920, 1080));
  imageSize->addItem("3840x2160", QSize(3840, 2160));
  imageSize->setValidator(new QRegExpValidator(
      QRegExp(R"([1-9]\d{0,5}\s*x\s*[1-9]\d{0,5})"), this));
  connect(imageSize, SIGNAL(activated(int)), this,
          SLOT(fixImageSizeQVariant(int)));
  controlGroupLayout->addRow(tr("&Image Size"), imageSize);

  auto seed = new QSpinBox();
  seed->setRange(std::numeric_limits<int>::min() + 1,
                 std::numeric_limits<int>::max() - 1);
  seed->setValue(42);
  seed->setSingleStep(1000);
  seed->setToolTip(tr("Random seed value"));
  controlGroupLayout->addRow(tr("&Seed"), seed);

  auto samples = new QSpinBox();
  samples->setRange(1, std::numeric_limits<int>::max() - 1);
  samples->setValue(30000);
  samples->setSingleStep(1000);
  samples->setToolTip(tr("Number of passes for each variation [1, +oo]"));
  controlGroupLayout->addRow(tr("&Samples"), samples);

  auto iterations = new QSpinBox();
  iterations->setRange(1, 1024);
  iterations->setValue(1024);
  iterations->setSingleStep(100);
  iterations->setToolTip(tr("Number of iterations per sample [1, 1024]"));
  controlGroupLayout->addRow(tr("&Iterations"), iterations);

  auto symmetry = new QSpinBox();
  symmetry->setRange(1, 72);
  symmetry->setValue(1);
  symmetry->setToolTip(tr("Number of rotations [1, 72]"));
  controlGroupLayout->addRow(tr("&Symmetry"), symmetry);

  auto supersampling = new QSpinBox();
  supersampling->setRange(1, 16);
  supersampling->setValue(1);
  supersampling->setToolTip(tr("Render image in bigger resolution [1, 16]"));
  controlGroupLayout->addRow(tr("&Supersampling"), supersampling);

  auto addVariation = new QPushButton(tr("Add"));
  auto removeVariation = new QPushButton(tr("Remove"));
  auto variationLayout = new QHBoxLayout();
  variationLayout->addWidget(addVariation);
  variationLayout->addWidget(removeVariation);
  controlGroupLayout->addRow(tr("Varations"), variationLayout);

  auto variationList = new QTreeView();
  variationList->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  auto sp = variationList->sizePolicy();
  sp.setVerticalStretch(1);
  variationList->setSizePolicy(sp);
  controlGroupLayout->addRow(variationList);

  auto loadVariations = new QPushButton(tr("Load..."));
  auto saveVariations = new QPushButton(tr("Save..."));
  auto fileOpLayout = new QHBoxLayout();
  fileOpLayout->addWidget(loadVariations);
  fileOpLayout->addWidget(saveVariations);
  controlGroupLayout->addRow(fileOpLayout);

  // set central widget
  auto splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(viewScroll);
  splitter->addWidget(controlGroup);
  splitter->setStretchFactor(0, 1);
  splitter->setStretchFactor(1, 0);
  splitter->setCollapsible(0, false);
  splitter->setCollapsible(1, false);

  setCentralWidget(splitter);
}

void MainWindow::fixImageSizeQVariant(int index) const {
  auto data = imageSize->itemData(index);
  if (!data.isValid()) {
    auto text = imageSize->itemText(index);
    auto s = text.split(QRegExp(R"(\s*x\s*)"));
    imageSize->setItemData(index, QSize{s[0].toInt(), s[1].toInt()});
  }
}
