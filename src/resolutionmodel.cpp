#include "resolutionmodel.hpp"

#include <iostream>

ResolutionModel::ResolutionModel(QObject *parent)
    : QAbstractListModel{parent} {}

void ResolutionModel::setResolutionModel(std::vector<QSize> resolutions) {
  resolutionList = std::move(resolutions);
}

int ResolutionModel::rowCount(const QModelIndex & /* parent */) const {
  return static_cast<int>(resolutionList.size());
}

QVariant ResolutionModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid() ||
      (index.row() >= static_cast<int>(resolutionList.size()) ||
       index.row() < 0)) {
    return {};
  }

  QSize size = resolutionList[static_cast<unsigned long long>(index.row())];
  if (role == Qt::DisplayRole) {
    return QString("%1x%2").arg(size.width()).arg(size.height());
  } else if (role == Qt::UserRole) {
    return size;
  }
  return QVariant();
}
