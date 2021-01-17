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

#include <DO/Sara/Graphics/DerivedQObjects/RotationSliders.hpp>

#include <QtDebug>
#include <QGridLayout>
#include <QLabel>
#include <QSlider>


namespace DO { namespace Sara {

  RotationSliders::RotationSliders(QWidget* parent)
    : QWidget{parent}
    , m_yawSlider{new QSlider{Qt::Horizontal}}
    , m_pitchSlider{new QSlider{Qt::Horizontal}}
    , m_rollSlider{new QSlider{Qt::Horizontal}}
  {
    m_yawSlider->setRange(-180, 180);
    m_yawSlider->setTickPosition(QSlider::TicksBelow);
    m_yawSlider->setTickInterval(1);

    m_pitchSlider->setRange(-90, 90);
    m_pitchSlider->setTickPosition(QSlider::TicksBelow);
    m_pitchSlider->setTickInterval(1);

    m_rollSlider->setRange(-180, 180);
    m_rollSlider->setTickPosition(QSlider::TicksBelow);
    m_rollSlider->setTickInterval(1);

    auto yawLabel = new QLabel{"Yaw (About Z)"};
    auto yawAngle = new QLabel{};
    yawAngle->setNum(m_yawSlider->value());

    auto pitchLabel = new QLabel{"Pitch (About Y)"};
    auto pitchAngle = new QLabel{};
    pitchAngle->setNum(m_pitchSlider->value());

    auto rollLabel = new QLabel{"Roll (About X)"};
    auto rollAngle = new QLabel{};
    rollAngle->setNum(m_rollSlider->value());

    auto layout = new QGridLayout;
    layout->addWidget(yawLabel, 0, 0);
    layout->addWidget(yawAngle, 0, 1);
    layout->addWidget(m_yawSlider, 0, 2);

    layout->addWidget(pitchLabel, 1, 0);
    layout->addWidget(pitchAngle, 1, 1);
    layout->addWidget(m_pitchSlider, 1, 2);

    layout->addWidget(rollLabel, 2, 0);
    layout->addWidget(rollAngle, 2, 1);
    layout->addWidget(m_rollSlider, 2, 2);

    setLayout(layout);
    setWindowTitle("Euler Angles");
    show();

    connect(m_yawSlider, &QSlider::valueChanged, this, &RotationSliders::notifyNewAngles);
    connect(m_yawSlider, &QSlider::valueChanged, yawAngle, qOverload<int>(&QLabel::setNum));

    connect(m_pitchSlider, &QSlider::valueChanged, this, &RotationSliders::notifyNewAngles);
    connect(m_pitchSlider, &QSlider::valueChanged, pitchAngle, qOverload<int>(&QLabel::setNum));

    connect(m_rollSlider, &QSlider::valueChanged, this, &RotationSliders::notifyNewAngles);
    connect(m_rollSlider, &QSlider::valueChanged, rollAngle, qOverload<int>(&QLabel::setNum));
  }

  void RotationSliders::notifyNewAngles(int) {
    const auto yaw = m_yawSlider->value();
    const auto pitch = m_pitchSlider->value();
    const auto roll = m_rollSlider->value();
    emit sendNewAngles(yaw, pitch, roll);
  }


}}  // namespace DO::Sara
