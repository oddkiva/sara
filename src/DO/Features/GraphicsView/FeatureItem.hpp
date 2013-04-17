/*
 * =============================================================================
 *
 *       Filename:  FeatureItem.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  01/08/2011 03:05:04
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  David OK (DO), david.ok8@gmail.com
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_AFFINECOVARIANTFEATURES_FEATUREITEM_H
#define DO_AFFINECOVARIANTFEATURES_FEATUREITEM_H

#include <QtGui>
#include <QGraphicsItem>
#include <QGraphicsEllipseItem>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>
#include <DO/Core.hpp>

namespace DO {

    template <typename Feature>
    struct FeatureItemSelector
    {
        class FeatureItem;
        class MatchItem;

        class FeatureItem : public QGraphicsEllipseItem
        {
        public:
            FeatureItem(const Feature& feature,
                        const QColor& color = Qt::yellow,
                        const QColor& colorOnHover = Qt::red,
                        qreal penWidth = 2)
              : QGraphicsEllipseItem(computeBoundingRect(feature)),
                feature_(feature), color_(color), colorOnHover_(colorOnHover),
                penWidth_(penWidth)
            {
				setPos(feature_.x(), feature_.y());
                //setCacheMode(ItemCoordinateCache);
                setAcceptHoverEvents(true);
				setOpacity(0.7);
				/*setFlags(ItemSendsGeometryChanges |
                         ItemSendsScenePositionChanges);
                setZValue(10);*/
				setPen(QPen(color_, 1));
            }

            inline
            const Feature& feature() const
            {
                return feature_;
            }

            inline
            QList<MatchItem*>& matchList()
            {
                return matchList_;
            }

            void addMatch(MatchItem* match)
            {
                matchList_ << match;
                match->adjust();
            }

        protected:
            void paint(QPainter* painter,
                       const QStyleOptionGraphicsItem* option,
                       QWidget* widget)
            {
				const qreal lod = option->
					levelOfDetailFromTransform(painter->worldTransform());
                
				if (lod < 0.25)
					return;
                
                QGraphicsEllipseItem::paint(painter, option, widget);
            }
            
            QVariant itemChange(GraphicsItemChange change,
                                const QVariant& value)
            {
                switch (change)
                {
                    case ItemScenePositionHasChanged:
                        foreach (MatchItem* match, matchList_)
                            match->adjust();
                        break;
                    default:
                        break;
                }

				return QGraphicsItem::itemChange(change, value);
            }

            void hoverEnterEvent(QGraphicsSceneHoverEvent* event)
            {
                QPen pen(colorOnHover_, penWidth_+1);
                foreach (MatchItem* m, matchList())
                {
                    m->show();
                    m->source()->setPen(pen);
                    m->target()->setPen(pen);
                    m->source()->update();
                    m->target()->update();
                }
            }

            void hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
            {
                QPen pen(color_);
                foreach (MatchItem* m, matchList())
                {
                    m->hide();
                    m->source()->setPen(pen);
                    m->target()->setPen(pen);
                    m->source()->update();
                    m->target()->update();
                }
            }

		private:
			QRectF computeBoundingRect(const Feature& f)
			{
				typedef typename Feature::Matrix2x2 Matrix2x2;
				Eigen::JacobiSVD<Matrix2x2> svd(f.scaleMatrix());
				const qreal a = qreal(1)/std::sqrt(svd.singularValues()(0));
				const qreal b = qreal(1)/std::sqrt(svd.singularValues()(1));
				return QRectF(-a, -b, 2*a, 2*b);
			}

        private:
            const Feature& feature_;
            QList<MatchItem *> matchList_;
            QColor color_;
            QColor colorOnHover_;
            qreal  penWidth_;
        };

        class MatchItem : public QGraphicsItem
        {
        public:
            MatchItem(FeatureItem* source, FeatureItem* target,
                      qreal arrowSize = qreal(5),
                      const QColor& color = Qt::red,
                      const QColor& colorOnHover = Qt::blue)
              : source_(source), target_(target), arrowSize_(5),
                color_(color), colorOnHover_(colorOnHover), currentColor_(color)
            {
				source->addMatch(this);
                target->addMatch(this);
                setAcceptedMouseButtons(0);
				//setAcceptHoverEvents(true);
                setVisible(false);
                setFlags(ItemIsFocusable);
                setOpacity(0.5);
                setZValue(qreal(0));
                adjust();
            }

            inline FeatureItem* source() const { return source_; }
            inline FeatureItem* target() const { return target_; }
            
            void adjust()
            {
                if (!source_ || !target_)
                    return;

                QLineF line(mapFromItem(source_, 0, 0), 
                            mapFromItem(target_, 0, 0));
                qreal length = line.length();

                prepareGeometryChange();

                if (length > qreal(20.))
                {
                    QPointF edgeOffset( (line.dx()*5) / length,
                                        (line.dy()*5) / length);
                    sourcePoint_ = line.p1() + edgeOffset;
                    targetPoint_ = line.p2() - edgeOffset;
                }
                else
                    sourcePoint_ = targetPoint_ = line.p1();              
            }

            inline
            void mark()
            {
                currentColor_ = colorOnHover_;
                update();
            }

            inline
            void unmark()
            {
                currentColor_ = color_;
                update();
            }

        protected:
            QRectF boundingRect() const
            {
                if (!source_ || !target_)
                    return QRectF();

                qreal penWidth = 1;
                qreal extra = (penWidth + arrowSize_) / 2.0;

                return QRectF(sourcePoint_,
                              QSizeF(targetPoint_.x() - sourcePoint_.x(),
                                     targetPoint_.y() - sourcePoint_.y()))
                    .normalized()
                    .adjusted(-extra, -extra, extra, extra);
            }

            void paint(QPainter* painter,
                       const QStyleOptionGraphicsItem* option,
                       QWidget* widget)

            {
                if (!source_ || !target_)
                    return;

                QLineF line(sourcePoint_, targetPoint_);
                if (qFuzzyCompare(line.length(), qreal(0.)))
                    return;

                // Draw the line itself
                painter->setPen(QPen(currentColor_, 2,
                                     Qt::SolidLine, Qt::RoundCap, 
                                     Qt::RoundJoin));
                painter->drawLine(line);

				const qreal Pi = pi<qreal>();

                // Draw the arrows
                double angle = std::atan2(line.dy(), line.dx());

                QPointF sourceArrowP1 = sourcePoint_
                                      + QPointF(cos(angle+Pi/12)*arrowSize_,
                                                sin(angle+Pi/12)*arrowSize_);
                QPointF sourceArrowP2 = sourcePoint_
                                      + QPointF(cos(angle-Pi/12)*arrowSize_,
                                                sin(angle-Pi/12)*arrowSize_);
                QPointF targetArrowP1 = targetPoint_
                                      + QPointF(cos(angle+Pi-Pi/12)*arrowSize_,
                                                sin(angle+Pi-Pi/12)*arrowSize_);
                QPointF targetArrowP2 = targetPoint_
                                      + QPointF(cos(angle+Pi+Pi/12)*arrowSize_,
                                                sin(angle+Pi+Pi/12)*arrowSize_);

				painter->drawPolygon(QPolygonF() << line.p1() <<
									 sourceArrowP1 << sourceArrowP2);
                painter->drawPolygon(QPolygonF() << line.p2() <<
									 targetArrowP1 << targetArrowP2);
            }

//            inline
//            void hoverEnterEvent(QGraphicsSceneHoverEvent* event)
//            {
//                mark();
//            }
//
//            inline
//            void hoverLeaveEvent(QGraphicsSceneHoverEvent* event)
//            {
//                unmark();
//            }

        private:
            FeatureItem *source_;
            FeatureItem *target_;
            QPointF sourcePoint_;
            QPointF targetPoint_;
            qreal arrowSize_;
            QColor color_;
            QColor colorOnHover_;
            QColor currentColor_;
        };
    };

} /* namespace DO */

#endif /* DO_AFFINECOVARIANTFEATURES_FEATUREITEM_H */
