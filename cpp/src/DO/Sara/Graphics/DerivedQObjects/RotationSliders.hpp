// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <QWidget>


class QSlider;

namespace DO { namespace Sara {

  class DO_SARA_EXPORT RotationSliders : public QWidget
  {
    Q_OBJECT

  public:
    RotationSliders(QWidget* parent = nullptr);

  signals:
    void sendNewAngles(int, int, int);

  public slots:
    void notifyNewAngles(int);

  private:
    QSlider *m_yawSlider = nullptr;
    QSlider *m_pitchSlider = nullptr;
    QSlider *m_rollSlider = nullptr;
  };


}}  // namespace DO::Sara
