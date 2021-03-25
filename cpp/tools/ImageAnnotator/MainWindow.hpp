// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2010-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
#ifndef DO_IMAGEANNOTATOR_MAINWINDOW_HPP
#define DO_IMAGEANNOTATOR_MAINWINDOW_HPP

#include <QGraphicsItem>
#include <QGraphicsView>
#include <QListWidget>
#include <QMainWindow>
#include <QPainter>
#include <QPixmap>
#include <QWidget>

// ========================================================================== //
// File browser class
class FileBrowser : public QListWidget
{
  Q_OBJECT
public:
  FileBrowser(const QStringList& filters, QWidget* parent = 0);
  void setDir(const QString& path);
  QString getBasePath() const
  {
    return basePath;
  }

signals:
  void picked(const QString& fileName);

private slots:
  void selectItem(QListWidgetItem* item);

private:
  QStringList nameFilters;
  QString basePath;
};

// ========================================================================== //
// Painting Widget
class AnnotatingWidget : public QWidget
{
  Q_OBJECT

public:
  AnnotatingWidget(QWidget* parent = 0);

  void setPixmap(const QPixmap& pixmap);
  void setBBoxes(const QList<QRectF>& loadedBBoxes);
  const QList<QRectF>& getBBoxes()
  {
    return bboxes;
  }

public slots:
  void clear();
  void undo();
  void redo();
  bool save(const QString& fn);

protected:
  void mousePressEvent(QMouseEvent* e);
  void mouseMoveEvent(QMouseEvent* e);
  void mouseReleaseEvent(QMouseEvent* e);
  void paintEvent(QPaintEvent* e);

private:
  QPainter painter;
  QPixmap image;

  QList<QRectF>::iterator lastBBox;
  QList<QRectF> bboxes;
};

// ========================================================================== //
// Graphics Items
class Point;
class Line;
class Quad;

class Point : public QGraphicsItem
{
public:
  enum
  {
    Type = UserType + 1
  };
  int type() const
  {
    return Type;
  }

  Point(const QPointF& pos);
  Quad* quad() const
  {
    return qgraphicsitem_cast<Quad*>(parentItem());
  }

  void addLine(Line* line);
  QList<Line*> lines() const;

  QRectF boundingRect() const;
  QPainterPath shape() const;
  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option,
             QWidget* widget);

protected:
  QVariant itemChange(GraphicsItemChange change, const QVariant& value);

  void mousePressEvent(QGraphicsSceneMouseEvent* event);
  void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

private:
  QList<Line*> lineList;
};

class Line : public QGraphicsItem
{
public:
  enum
  {
    Type = UserType + 2
  };
  int type() const
  {
    return Type;
  }

  Line(Point* sourceNode, Point* destNode);

  Point* sourceNode() const;
  Point* destNode() const;

  void adjust();

protected:
  QRectF boundingRect() const;
  void paint(QPainter* painter, const QStyleOptionGraphicsItem* option,
             QWidget* widget);

private:
  Point *source, *dest;
  QPointF sourcePoint;
  QPointF destPoint;
};

class Quad : public QGraphicsPolygonItem
{
public:
  Quad(Point* points[4]);
  ~Quad();
  bool operator==(const Quad& quad) const;
  Point* const* points() const
  {
    return p;
  }
  void adjust();
  void print() const;

  enum
  {
    Type = UserType + 3
  };
  int type() const
  {
    return Type;
  }

  QVariant itemChange(GraphicsItemChange change, const QVariant& value);

private:
  Point* p[4];
};

// ========================================================================== //
// Graphics View Widget
class QGraphicsPixmapItem;

class GraphicsAnnotator : public QGraphicsView
{
  Q_OBJECT
public:
  GraphicsAnnotator(int width, int height, QWidget* parent = 0);
  void setPixmap(const QPixmap& pixmap);
  void setQuads(const QList<QPolygonF>& quads);
  void clearQuads();
  bool save(const QString& fn);

private:
  // Zoom in/out the scene view
  void scaleView(qreal scaleFactor);

protected:
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void wheelEvent(QWheelEvent* event);
  void keyPressEvent(QKeyEvent* event);

private:
  int counter;
  bool creatingRect;
  Point* quad[4];
  QList<Quad*> quads;
  QGraphicsPixmapItem* pix;
};

// ========================================================================== //
// Main window class
class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  MainWindow(QWidget* parent = 0);

private slots:
  void displayImageAndBBoxes(const QString& filepath);
  void openFolder();
  void saveQuads();
  void about();

private:
  void createDockableFileBrowser();
  void createDockableQuadFileBrowser();
  void createCentralGraphicsView();
  void createFileActions();
  void createQuadActions();
  void createHelpActions();
  void createConnections();

private:
  QString currentFilePath;
  GraphicsAnnotator* annotator;
  FileBrowser* fileBrowser;
  FileBrowser* quadFileBrowser;
};

#endif /* DO_IMAGEANNOTATOR_MAINWINDOW_HPP */
