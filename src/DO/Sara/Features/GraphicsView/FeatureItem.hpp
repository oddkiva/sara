// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_AFFINECOVARIANTFEATURES_FEATUREITEM_HPP
#define DO_SARA_AFFINECOVARIANTFEATURES_FEATUREITEM_HPP

#include <cmath>

#include <QGraphicsEllipseItem>
#include <QGraphicsItem>
#include <QtGui>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <DO/Sara/Core.hpp>


namespace DO { namespace Sara {

  class OERegionItem;
  class MatchItem;

  class OERegionItem : public QGraphicsEllipseItem
  {
  public:
    OERegionItem(const OERegion& feature,
                 const QColor& color = Qt::yellow,
                 const QColor& colorOnHover = Qt::red,
                 qreal penWidth = 2);
    const Feature& feature() const { return feature_; }
    QList<MatchItem *>& matchList() { return matchList_; }
    void addMatch(MatchItem* match);

  protected:
    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem* option,
               QWidget* widget);
    QVariant itemChange(GraphicsItemChange change,
                        const QVariant& value);
    void hoverEnterEvent(QGraphicsSceneHoverEvent* event);
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event);

  private:
    QRectF computeBoundingRect(const Feature& f);

  private:
    const OERegion& feature_;
    QList<MatchItem *> matchList_;
    QColor color_;
    QColor colorOnHover_;
    qreal  penWidth_;
  };

  class MatchItem : public QGraphicsItem
  {
  public:
    MatchItem(OERegionItem* source, OERegionItem* target,
              qreal arrowSize = qreal(5),
              const QColor& color = Qt::red,
              const QColor& colorOnHover = Qt::blue);
    OERegionItem* source() const { return source_; }
    OERegionItem* target() const { return target_; }
    void adjust();
    void mark();
    void unmark();

  protected:
    QRectF boundingRect() const;
    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem* option,
               QWidget* widget);
    //inline
    //void hoverEnterEvent(QGraphicsSceneHoverEvent* event)
    //{ mark(); }
    //
    //inline
    //void hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
    //{ unmark(); }

  private:
    OERegionItem *source_;
    OERegionItem *target_;
    QPointF sourcePoint_;
    QPointF targetPoint_;
    qreal arrowSize_;
    QColor color_;
    QColor colorOnHover_;
    QColor currentColor_;
  };

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_AFFINECOVARIANTFEATURES_FEATUREITEM_HPP */
