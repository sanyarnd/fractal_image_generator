#ifndef RESOLUTIONMODEL_HPP
#define RESOLUTIONMODEL_HPP

#include <QAbstractListModel>
#include <QSize>
#include <vector>

class ResolutionModel : public QAbstractListModel {
  Q_OBJECT

public:
  explicit ResolutionModel(QObject *parent = nullptr);

  void setResolutionModel(std::vector<QSize> resolutions);

  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  QVariant data(const QModelIndex &index,
                int role = Qt::DisplayRole) const override;

private:
  std::vector<QSize> resolutionList;
};

#endif // RESOLUTIONMODEL_HPP
