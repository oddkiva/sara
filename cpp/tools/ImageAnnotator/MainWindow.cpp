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

#include "MainWindow.hpp"

#include <QGraphicsEllipseItem>
#include <QGraphicsItemGroup>
#include <QtGui>
#include <QtOpenGL>
#include <QDockWidget>
#include <QFileDialog>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QOpenGLWidget>
#include <QSet>
#include <QStatusBar>
#include <QToolBar>

#include <algorithm>


// ========================================================================== //
// File browser class
FileBrowser::FileBrowser(const QStringList& filters, QWidget* parent)
  : QListWidget(parent)
  , nameFilters(filters)
{
  setDir(QStandardPaths::writableLocation(QStandardPaths::DesktopLocation));
  connect(this, SIGNAL(itemActivated(QListWidgetItem*)), this,
          SLOT(selectItem(QListWidgetItem*)));
  connect(this, SIGNAL(itemClicked(QListWidgetItem*)), this,
          SLOT(selectItem(QListWidgetItem*)));
}

void FileBrowser::setDir(const QString& path)
{
  QDir dir(path);
  dir.setNameFilters(nameFilters);
  dir.setSorting(QDir::DirsFirst);
  dir.setFilter(QDir::NoDot | QDir::AllEntries | QDir::AllDirs);
  if (!dir.isReadable())
    return;
  clear();

  /*QFileInfoList fileInfoList( dir.entryInfoList() );
  QFileInfoList::const_iterator f;
  for (f = fileInfoList.begin(); f != fileInfoList.end(); ++f)
  {
    if (!f->isFile())
    {
     addItem(f->fileName());
    }
    else
    {
      QListWidgetItem *item = new QListWidgetItem(
        QIcon(f->filePath()),
        f->fileName() );
      addItem(item);
    }
  }*/
  addItems(dir.entryList());

  basePath = dir.canonicalPath();
}

void FileBrowser::selectItem(QListWidgetItem* item)
{
  QString path = basePath + "/" + item->text();
  if (QFileInfo(path).isDir())
    setDir(path);
  else
    emit picked(path);
}

// ========================================================================== //
// Painting Widget
AnnotatingWidget::AnnotatingWidget(QWidget* parent)
  : QWidget(parent)
{
}

void AnnotatingWidget::setPixmap(const QPixmap& pixmap)
{
  image = pixmap;
  resize(pixmap.width(), pixmap.height());
  bboxes.clear();
  lastBBox = bboxes.end();
  update();
}

void AnnotatingWidget::setBBoxes(const QList<QRectF>& loadedBBoxes)
{
  bboxes = loadedBBoxes;
  lastBBox = bboxes.end();
  update();
}

void AnnotatingWidget::clear()
{
  bboxes.clear();
  lastBBox = bboxes.end();
  update();
}

void AnnotatingWidget::undo()
{
  if (lastBBox != bboxes.begin())
    --lastBBox;
  update();
}

void AnnotatingWidget::redo()
{
  if (lastBBox != bboxes.end())
    ++lastBBox;
  update();
}

bool AnnotatingWidget::save(const QString& fn)
{
  QFile file(fn);
  if (bboxes.empty())
    return false;
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    return false;
  QTextStream out(&file);
  QList<QRectF>::const_iterator b = bboxes.begin();
  for (; b != lastBBox; ++b)
    out << b->topLeft().x() << " " << b->topLeft().y() << " "
        << b->bottomRight().x() << " " << b->bottomRight().y() << "\n";
  file.close();
  return true;
}

void AnnotatingWidget::mousePressEvent(QMouseEvent* e)
{
  if (e->button() == Qt::LeftButton)
  {
    bboxes.erase(lastBBox, bboxes.end());

    // qDebug() << "mouse pressed";
    bboxes.push_back(QRectF());
    lastBBox = bboxes.end();

#if QT_VERSION_MAJOR == 6
    const auto pos = e->position();
#else
    const auto pos = e->localPos();
#endif
    bboxes.back().setTopLeft(pos);
    bboxes.back().setBottomRight(pos);
    // qDebug() << bboxes.back();
    update();
  }
}

void AnnotatingWidget::mouseMoveEvent(QMouseEvent* e)
{
#if QT_VERSION_MAJOR == 6
    const auto pos = e->position();
#else
    const auto pos = e->localPos();
#endif
  // qDebug() << "mouse moved";
  bboxes.back().setBottomRight(pos);
  // qDebug() << bboxes.back();
  update();
}

void AnnotatingWidget::mouseReleaseEvent(QMouseEvent* e)
{
#if QT_VERSION_MAJOR == 6
    const auto pos = e->position();
#else
    const auto pos = e->localPos();
#endif
  // qDebug() << "mouse released";
  bboxes.back().setBottomRight(pos);
  update();
  qDebug() << bboxes.back();
  lastBBox = bboxes.end();
}

void AnnotatingWidget::paintEvent(QPaintEvent*)
{
  QPainter painter(this);
  QPen pen(Qt::yellow, 2);
  pen.setCapStyle(Qt::RoundCap);
  painter.setPen(pen);
  painter.drawPixmap(0, 0, image);
  for (QList<QRectF>::const_iterator b = bboxes.begin(); b != lastBBox; ++b)
    painter.drawRect(*b);
}

// ========================================================================== //
// Graphics Items
Point::Point(const QPointF& pos)
{
  setFlags(ItemIsMovable | ItemIsSelectable);
  setFlag(ItemSendsGeometryChanges);
  setCacheMode(DeviceCoordinateCache);
  setZValue(1.);
  setPos(pos);
  setOpacity(0.8);
}

void Point::addLine(Line* line)
{
  lineList << line;
  line->adjust();
}

QList<Line*> Point::lines() const
{
  return lineList;
}

QRectF Point::boundingRect() const
{
  qreal adjust = 2;
  return QRectF(-10 - adjust, -10 - adjust, 23 + adjust, 23 + adjust);
}

QPainterPath Point::shape() const
{
  QPainterPath path;
  path.addEllipse(-10, -10, 20, 20);
  return path;
}

void Point::paint(QPainter* painter, const QStyleOptionGraphicsItem* option,
                  QWidget*)
{
  bool selected = isSelected();
  if (parentItem())
    selected = selected || parentItem()->isSelected();
  QColor c(selected ? Qt::red : Qt::yellow);
  if (option->state & QStyle::State_Sunken)
    c = c.lighter(120);

  painter->setPen(QPen(c, 1, Qt::SolidLine, Qt::FlatCap, Qt::MiterJoin));
  painter->drawEllipse(-3, -3, 6, 6);
}

QVariant Point::itemChange(GraphicsItemChange change, const QVariant& value)
{
  Quad* q;
  switch (change)
  {
  case ItemPositionHasChanged:
    foreach (Line* line, lineList)
      line->adjust();
    q = quad();
    if (q)
      q->adjust();
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

void Point::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
  update();
  QGraphicsItem::mousePressEvent(event);
}

void Point::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
  update();
  QGraphicsItem::mouseReleaseEvent(event);
}

Line::Line(Point* sourceNode, Point* destNode)
{
  setAcceptedMouseButtons(Qt::NoButton);
  setFlags(ItemIsSelectable);
  source = sourceNode;
  dest = destNode;
  source->addLine(this);
  dest->addLine(this);
  setOpacity(0.8);
  adjust();
}

Point* Line::sourceNode() const
{
  return source;
}

Point* Line::destNode() const
{
  return dest;
}

void Line::adjust()
{
  if (!source || !dest)
    return;

  QLineF line(mapFromItem(source, 0, 0), mapFromItem(dest, 0, 0));

  prepareGeometryChange();

  sourcePoint = line.p1();
  destPoint = line.p2();
}

QRectF Line::boundingRect() const
{
  if (!source || !dest)
    return QRectF();

  return QRectF(sourcePoint, QSizeF(destPoint.x() - sourcePoint.x(),
                                    destPoint.y() - sourcePoint.y()))
      .normalized();
}

void Line::paint(QPainter* painter, const QStyleOptionGraphicsItem*, QWidget*)
{
  if (!source || !dest)
    return;

  QLineF line(sourcePoint, destPoint);
  if (qFuzzyCompare(line.length(), qreal(0.)))
    return;

  bool selected = isSelected();
  if (parentItem())
    selected = selected || parentItem()->isSelected();
  QColor c(selected ? Qt::red : Qt::yellow);
  painter->setPen(QPen(c, 1, Qt::SolidLine, Qt::RoundCap, Qt::MiterJoin));
  painter->drawLine(line);
}

Quad::Quad(Point* points[4])
{
  setFlags(ItemIsSelectable);
  setPen(Qt::NoPen);
  for (int i = 0; i < 4; ++i)
  {
    p[i] = points[i];
    p[i]->setParentItem(this);
    foreach (Line* line, p[i]->lines())
      line->setParentItem(this);
  }
  setBrush(QColor(Qt::yellow).lighter());
  setOpacity(0.5);
  adjust();
}

Quad::~Quad()
{
  // Destroy the lines.
  QList<Line*> lines;
  for (int i = 0; i < 4; ++i)
    lines << p[i]->lines();

  // Eliminate duplicates.
#if QT_VERSION_MAJOR == 6
  lines = QSet<Line *>(lines.begin(), lines.end()).values();
#else
  lines = lines.toSet().toList();
#endif

  // Destroy the lines.
  for (int i = 0; i < lines.size(); ++i)
    delete lines[i];

  // Destroy the points.
  for (int i = 0; i < 4; ++i)
    delete p[i];
}

bool Quad::operator==(const Quad& quad) const
{
  for (int i = 0; i < 4; ++i)
    if (p[i] != quad.p[i])
      return false;
  return true;
}

void Quad::adjust()
{
  QPolygonF poly(4);
  for (int i = 0; i < 4; ++i)
    poly[i] = mapFromItem(p[i], 0, 0);
  setPolygon(poly);

  prepareGeometryChange();
}

void Quad::print() const
{
  qDebug() << "Quad";
  for (int i = 0; i < 4; ++i)
    qDebug() << i << " " << p[i]->pos();
}

QVariant Quad::itemChange(GraphicsItemChange change, const QVariant& value)
{
  switch (change)
  {
  case ItemPositionHasChanged:
    adjust();
    break;
  default:
    break;
  };

  return QGraphicsItem::itemChange(change, value);
}

// ========================================================================== //
// Graphics View Widget
GraphicsAnnotator::GraphicsAnnotator(int, int, QWidget* parent)
  : QGraphicsView(parent)
  , counter(0)
  , creatingRect(false)
{
  setViewport(new QOpenGLWidget);
  setTransformationAnchor(AnchorUnderMouse);
  setRenderHints(QPainter::Antialiasing);
  setDragMode(RubberBandDrag);

  setScene(new QGraphicsScene);

  pix = new QGraphicsPixmapItem;
  pix->setTransformationMode(Qt::SmoothTransformation);
  scene()->addItem(pix);
}

void GraphicsAnnotator::setPixmap(const QPixmap& pixmap)
{
  pix->setPixmap(pixmap);
  setSceneRect(pixmap.rect());
}

void GraphicsAnnotator::setQuads(const QList<QPolygonF>& newQuads)
{
  if (!quads.empty())
    clearQuads();

  qDebug() << "\nCreating quads";
  QList<QPolygonF>::const_iterator q;
  for (q = newQuads.begin(); q != newQuads.end(); ++q)
  {
    Point* pts[4];
    for (int i = 0; i < 4; ++i)
      pts[i] = new Point((*q)[i]);

    Quad* quad = new Quad(pts);
    scene()->addItem(quad);
    quads.push_back(quad);
  }

  qDebug() << "Num quads: " << quads.size();
  qDebug() << "Num graphic items: " << scene()->items().size() << "\n";
}

void GraphicsAnnotator::clearQuads()
{
  qDebug() << "\nClearing quads";
  for (QList<Quad*>::iterator q = quads.begin(); q != quads.end(); ++q)
    delete *q;
  quads.clear();

  qDebug() << "Num quads: " << quads.size();
  qDebug() << "Num graphic items: " << scene()->items().size() << "\n";
}

bool GraphicsAnnotator::save(const QString& fn)
{
  QFile file(fn);
  if (quads.empty())
    return false;
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    return false;
  QTextStream out(&file);

  out << quads.size() << "\n";
  int poly = 0;
  for (QList<Quad*>::const_iterator q = quads.begin(); q != quads.end(); ++q)
  {
    qDebug() << "Saving poly #" << poly;
    ++poly;
    QPolygonF poly = (*q)->polygon();

    for (int i = 0; i < 4; ++i)
    {
      out << poly[i].x() << " " << poly[i].y() << " ";
      qDebug() << i << "\t" << poly[i].x() << " " << poly[i].y();
    }

    out << "\n";
  }
  file.close();
  return true;
}

void GraphicsAnnotator::scaleView(qreal scaleFactor)
{
  scale(scaleFactor, scaleFactor);
}

void GraphicsAnnotator::mousePressEvent(QMouseEvent* event)
{
  if (event->button() == Qt::LeftButton &&
      event->modifiers() == Qt::AltModifier)
  {
    Point* newPoint = new Point(mapToScene(event->pos()));
    scene()->addItem(newPoint);

    if (counter == 0)
    {
      creatingRect = false;
      quad[0] = newPoint;
    }
    if (counter >= 1 && counter < 4)
    {
      quad[counter] = newPoint;
      scene()->addItem(new Line(quad[counter - 1], quad[counter]));
    }
    if (counter == 3)
    {
      scene()->addItem(new Line(quad[3], quad[0]));

      Quad* newQuad = new Quad(quad);
      scene()->addItem(newQuad);
      quads.push_back(newQuad);
      qDebug() << "\nAdded quad";
      quads.back()->print();
      qDebug() << "num quads: " << quads.size();
      creatingRect = false;
    }

    counter = (counter + 1) % 4;

    return;
  }

  QGraphicsView::mousePressEvent(event);
}

void GraphicsAnnotator::mouseReleaseEvent(QMouseEvent* event)
{
  if (event->buttons() == Qt::LeftButton &&
      event->modifiers() == Qt::AltModifier && creatingRect)
    creatingRect = false;
  QGraphicsView::mouseReleaseEvent(event);
}

void GraphicsAnnotator::mouseMoveEvent(QMouseEvent* event)
{
  if (event->buttons() == Qt::LeftButton &&
      event->modifiers() == Qt::AltModifier && counter == 1)
  {
    QPointF startPoint(quad[0]->pos());
    QPointF endPoint(mapToScene(event->pos()));

    if (!creatingRect)
    {
      creatingRect = true;

      quad[1] = new Point(QPointF(endPoint.x(), startPoint.y()));
      quad[2] = new Point(endPoint);
      quad[3] = new Point(QPointF(startPoint.x(), endPoint.y()));

      scene()->addItem(new Line(quad[0], quad[1]));
      scene()->addItem(new Line(quad[1], quad[2]));
      scene()->addItem(new Line(quad[2], quad[3]));
      scene()->addItem(new Line(quad[3], quad[0]));

      Quad* newQuad = new Quad(quad);
      scene()->addItem(newQuad);
      quads.push_back(newQuad);
      qDebug() << "\nAdded quad";
      quads.back()->print();
      qDebug() << "num quads: " << quads.size();
    }

    counter = 0;
  }

  if (event->buttons() == Qt::LeftButton &&
      event->modifiers() == Qt::AltModifier && creatingRect && counter == 0)
  {
    QPointF startPoint(quad[0]->pos());
    QPointF endPoint(mapToScene(event->pos()));
    quad[1]->setPos(QPointF(endPoint.x(), startPoint.y()));
    quad[2]->setPos(endPoint);
    quad[3]->setPos(QPointF(startPoint.x(), endPoint.y()));
  }

  QGraphicsView::mouseMoveEvent(event);
}

void GraphicsAnnotator::wheelEvent(QWheelEvent* event)
{
  if (event->modifiers() == Qt::ControlModifier)
    scaleView(pow(double(2), event->angleDelta().y() / 240.0));
  QGraphicsView::wheelEvent(event);
}

void GraphicsAnnotator::keyPressEvent(QKeyEvent* event)
{
  // Adjust view.
  if (event->key() == Qt::Key_F)
    fitInView(sceneRect(), Qt::KeepAspectRatio);

  // Delete the quads.
  if (event->key() == Qt::Key_Backspace || event->key() == Qt::Key_Delete)
  {
    // Get the selected quads.
    QList<Quad*> selectedQuads;
    QList<QGraphicsItem*> selectedItems = scene()->selectedItems();
    QList<QGraphicsItem*>::iterator item;
    for (item = selectedItems.begin(); item != selectedItems.end(); ++item)
    {
      Point* p = qgraphicsitem_cast<Point*>(*item);
      if (p && p->quad())
        selectedQuads << p->quad();

      Quad* q = qgraphicsitem_cast<Quad*>(*item);
      if (q)
        selectedQuads << q;
    }

    // Eliminate duplicates.
#if QT_VERSION_MAJOR == 6
    selectedQuads = QSet<Quad *>(selectedQuads.begin(),
                                 selectedQuads.end()).values();
#else
    selectedQuads = selectedQuads.toSet().toList();
#endif

    // Print the selected quads.
    qDebug() << "Selected quads";
    for (int i = 0; i < selectedQuads.size(); ++i)
      if (selectedQuads[i])
        selectedQuads[i]->print();

    // Destroy them.
    for (int i = 0; i < selectedQuads.size(); ++i)
    {
      auto q = std::find(quads.begin(), quads.end(), selectedQuads[i]);
      if (q != quads.end())
      {
        scene()->removeItem(*q);
        delete *q;
        quads.erase(q);
      }
      else
        qDebug() << "Error: could not retrieve quad";
    }
    qDebug() << "Remaining quads: " << quads.size();
    qDebug() << "Remaining graphic items: " << scene()->items().size() << "\n";
  }

  QGraphicsView::keyPressEvent(event);
}

// ========================================================================== //
// Main window class
MainWindow::MainWindow(QWidget* parent)
  : QMainWindow(parent)
{
  createDockableFileBrowser();
  createDockableQuadFileBrowser();
  createCentralGraphicsView();
  createFileActions();
  createQuadActions();
  createHelpActions();

  createConnections();

  resize(800, 600);
  setWindowTitle(tr("Image Annotator"));
  statusBar()->showMessage(tr("Ready"));
}

void MainWindow::displayImageAndBBoxes(const QString& somefilepath)
{
  qDebug() << "\nLoading image from path:\n" << somefilepath;
  statusBar()->showMessage("Loading image from path: " + somefilepath);
  currentFilePath = somefilepath;
  annotator->setPixmap(QPixmap(currentFilePath));

  annotator->clearQuads();
  QString QuadFolder(fileBrowser->getBasePath() + "/Quads/");
  QString QuadFile(QuadFolder + QFileInfo(currentFilePath).baseName() + ".txt");

  quadFileBrowser->setDir(fileBrowser->getBasePath() + "/Quads/");

  QFile file(QuadFile);
  if (!file.exists())
  {
    qDebug() << "\nThe following Quad file does not yet exist:\n" << QuadFile;
    statusBar()->showMessage("The following Quad file does not yet exist: " +
                             QuadFile);
    return;
  }

  if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    qDebug() << "\nCannot open " << QuadFile;
    return;
  }

  qDebug() << "\nLoading quads from file:\n" << QuadFile;
  statusBar()->showMessage("Loading quads from file: " + QuadFile);
  QList<QPolygonF> quads;
  QTextStream in(&file);

  int n;
  in >> n;

  for (int i = 0; i < n; ++i)
  {
    QPolygonF quad;
    double x, y;
    for (int j = 0; j < 4; ++j)
    {
      in >> x >> y;
      quad << QPointF(x, y);
    }
    quads.push_back(quad);
  }
  file.close();

  annotator->setQuads(quads);
}

void MainWindow::openFolder()
{
  QString dir = QFileDialog::getExistingDirectory(
      this, tr("Open folder"), "/home",
      QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  fileBrowser->setDir(dir);
}

void MainWindow::saveQuads()
{
  // Create BBox directory.
  const QString quadFolder(fileBrowser->getBasePath() + "/Quads");
  if (QDir().mkdir(quadFolder))
    fileBrowser->setDir(fileBrowser->getBasePath());
  QString basename(QFileInfo(currentFilePath).baseName());
  if (annotator->save(quadFolder + "/" + basename + ".txt"))
  {
    statusBar()->showMessage("Saved " + quadFolder + "/" + basename + ".txt");
    quadFileBrowser->setDir(quadFolder);
  }
  else
    statusBar()->showMessage("Cannot save " + quadFolder + "/" + basename +
                             ".txt");
}

void MainWindow::about()
{
  QMessageBox::about(
      this, tr("About Image Annotator"),
      tr("<p>The <b>Image Annotator</b> is a minimalist program that allows "
         "you to select BBoxes. "
         "Wait and see if more features will be added on the fly...</p>"));
}

void MainWindow::createDockableFileBrowser()
{
  // Dock file browser
  QDockWidget* browserDock = new QDockWidget(tr("Image File Browser"), this);
  addDockWidget(Qt::LeftDockWidgetArea, browserDock);

  // File browser widget itself
  QStringList filters;
  filters << "*.bmp"
          << "*.jpg"
          << "*.jpeg"
          << "*.png"
          << "*.tiff"
          << "*.tif"
          << "*.ppm";
  fileBrowser = new FileBrowser(filters);
  browserDock->setWidget(fileBrowser);
}

void MainWindow::createDockableQuadFileBrowser()
{
  QDockWidget* browserDock = new QDockWidget(tr("Quad File Browser"), this);
  addDockWidget(Qt::RightDockWidgetArea, browserDock);

  // File browser widget itself
  QStringList filters;
  filters << "*.txt";
  quadFileBrowser = new FileBrowser(filters);
  browserDock->setWidget(quadFileBrowser);
  quadFileBrowser->setDir(
      QStandardPaths::writableLocation(QStandardPaths::DesktopLocation) +
      "/Quads");
}

void MainWindow::createCentralGraphicsView()
{
  annotator = new GraphicsAnnotator(300, 300);
  annotator->setBackgroundRole(QPalette::Base);
  annotator->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  setCentralWidget(annotator);
}

void MainWindow::createFileActions()
{
  QMenu* fileMenu = new QMenu(tr("&File"), this);
  // Open folder act
  QAction* openFolderAct = new QAction(tr("Open folder..."), this);
  openFolderAct->setShortcut(tr("Ctrl+O"));
  connect(openFolderAct, SIGNAL(triggered()), this, SLOT(openFolder()));
  fileMenu->addAction(openFolderAct);
  // Separator
  fileMenu->addSeparator();
  // Quit act
  QAction* quitAct = new QAction(tr("Quit..."), this);
  quitAct->setShortcut(tr("Alt+F4"));
  connect(quitAct, SIGNAL(triggered()), this, SLOT(close()));
  fileMenu->addAction(quitAct);

  // Register menus
  menuBar()->addMenu(fileMenu);
}

void MainWindow::createQuadActions()
{
  QToolBar* drawToolBar = new QToolBar(tr("Drawing Tool Bar"), this);
  // Save
  QAction* saveAct = new QAction(tr("Save Quads"), this);
  saveAct->setShortcut(tr("S"));
  drawToolBar->addAction(saveAct);
  connect(saveAct, SIGNAL(triggered()), this, SLOT(saveQuads()));

  QMenu* actionMenu = new QMenu(tr("Quads"), this);
  actionMenu->addAction(saveAct);
  menuBar()->addMenu(actionMenu);

  addToolBar(drawToolBar);
}

void MainWindow::createHelpActions()
{
  QMenu* helpMenu = new QMenu(tr("Help"), this);
  // About act
  QAction* aboutAct = new QAction(tr("About"), this);
  connect(aboutAct, SIGNAL(triggered()), this, SLOT(about()));
  helpMenu->addAction(aboutAct);
  // About Qt act
  QAction* aboutQtAct = new QAction(tr("About Qt"), this);
  connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
  helpMenu->addAction(aboutQtAct);

  // Register menu
  menuBar()->addMenu(helpMenu);
}

void MainWindow::createConnections()
{
  // Refresh the image as the user picks a different image file in the file
  // browser.
  connect(fileBrowser, SIGNAL(picked(const QString&)), this,
          SLOT(displayImageAndBBoxes(const QString&)));
}
