// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_AFFINECOVARIANTFEATURES_VIEW_H
#define DO_AFFINECOVARIANTFEATURES_VIEW_H

#include <DO/Core.hpp>
#include <DO/Graphics.hpp>
#include "GraphicsView/FeatureItem.h"

namespace DO {

  /*!
    \ingroup Features
    @{
  */

  template <typename Keypoint>
  class ImageFeaturesItem : public QGraphicsPixmapItem
  {
    struct KeyShape
    {
      QPointF center;
      qreal radius;
      KeyShape(qreal x, qreal y, qreal r) : center(x, y), radius(r) {}
    };

  public:
    ImageFeaturesItem(const std::vector<Keypoint>& keys, QGraphicsItem *parent)
      : QGraphicsPixmapItem(0), shapes_(toKeyShapes(keys)),
      boundingRect_(parent->boundingRect()),
      shape_(parent->shape())
    {
      pen_.setWidth(2);
      pen_.setColor(Qt::yellow);
      setFlags(ItemSendsGeometryChanges |
        ItemSendsScenePositionChanges);

      setPos(0, 0);
      setOpacity(0.8);

      QPixmap pixmap(boundingRect().width(),
        boundingRect().height());
      pixmap.fill(Qt::transparent);
      QPainter p(&pixmap);
      p.setPen(pen_);
      p.setRenderHints(QPainter::Antialiasing | 
        QPainter::SmoothPixmapTransform);
      for(size_t i = 0; i != keys.size(); ++i)
        p.drawEllipse(keys[i].x(), keys[i].y(),
        keys[i].scale(), keys[i].scale());
      setPixmap(pixmap);
    }

  protected:
    void paint(QPainter *painter,
      const QStyleOptionGraphicsItem *option,
      QWidget *widget)
    {
      const qreal lod = option->
        levelOfDetailFromTransform(painter->worldTransform());
      qDebug() << "lod = " << lod;

      if (lod > 2)
      {
        qDebug("Careful detailed drawing");
        painter->setPen(pen_);

        QGraphicsView *v = scene()->views().front();
        QRect vRect = v->viewport()->rect();
        QRectF vRect_Scene = v->mapToScene(vRect).boundingRect();

        for(size_t i = 0; i != shapes_.size(); ++i)
        {
          if (!vRect_Scene.contains(mapToScene(shapes_[i].center)) )
            continue;

          painter->drawEllipse(shapes_[i].center, 
            shapes_[i].radius, shapes_[i].radius);
        }
        return;
      }

      if (lod < 0.1)
        return;

      qDebug("Default pixmap drawing");
      QGraphicsPixmapItem::paint(painter, option, widget);
      return;
    }

    QRectF boundingRect() const
    {
      return boundingRect_;
    }

    QPainterPath shape() const
    {
      return shape_;
    }

  private:
    std::vector<KeyShape> toKeyShapes(const std::vector<Keypoint>& keys)
    {
      std::vector<KeyShape> shapes;
      shapes.reserve(keys.size());
      for(size_t i = 0; i != keys.size(); ++i)
        shapes.push_back(KeyShape(keys[i].x(), keys[i].y(), keys[i].scale()));
      return shapes;
    }

  private:
    std::vector<KeyShape> shapes_;
    QPen pen_;
    QRectF boundingRect_;
    QPainterPath shape_;
  };


  template <typename FeatureOrKey>
  inline void putFeatureOnPixmapItem(const FeatureOrKey& key,
    QGraphicsPixmapItem *pixItem)
  {
    typedef FeatureItemSelector<FeatureOrKey> S;
    typedef typename S::FeatureItem FItem;
    QMetaObject::invokeMethod(scene(), "insertItem",
      Qt::QueuedConnection,
      Q_ARG(QGraphicsItem *, new FItem(key)),
      Q_ARG(QGraphicsItem *, pixItem));
  }

  template <typename FeatureOrKey>
  void simplePutFeaturesOnPixmapItem(const std::vector<FeatureOrKey>& keys,
    QGraphicsPixmapItem *pixItem)
  {
    for (size_t i = 0; i != keys.size(); ++i)
      putFeatureOnPixmapItem(keys[i], pixItem);
  }

  template <typename FeatureOrKey>
  void putFeaturesOnPixmapItem(const std::vector<FeatureOrKey>& keys,
    QGraphicsPixmapItem *pixItem)
  {
    typedef ImageFeaturesItem<FeatureOrKey> Features;
    QMetaObject::invokeMethod(scene(), "insertItem",
      Qt::QueuedConnection,
      Q_ARG(QGraphicsItem *,
      new Features(keys, pixItem)),
      Q_ARG(QGraphicsItem *, pixItem));
  }

  //! @file

} /* namespace DO */

#endif /* DO_AFFINECOVARIANTFEATURES_VIEW_H */
