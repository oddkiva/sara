// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <iostream>
#include <memory>

#include <Eigen/Core>


namespace DO::Sara {

  class CameraModel
  {
  public:
    template <typename Impl>
    CameraModel(Impl impl)
      : _self{new CameraModelImpl<Impl>(std::move(impl))}
    {
    }

    CameraModel(const CameraModel& c)
      : _self{c._self->copy()}
    {
    }

    CameraModel(CameraModel&& c) noexcept = default;

    CameraModel& operator=(const CameraModel& c)
    {
      auto tmp = CameraModel{c};
      *this = std::move(tmp);
      return *this;
    }

    CameraModel& operator=(CameraModel&& c) = default;

    friend auto project(const CameraModel& c, const Eigen::Vector3f& x)
        -> Eigen::Vector2f
    {
      return c._self->project(x);
    }

    friend auto backproject(const CameraModel& c, const Eigen::Vector2f& x)
        -> Eigen::Vector3f
    {
      return c._self->backproject(x);
    }

  private:
    struct CameraModelConcept
    {
      virtual ~CameraModelConcept() = default;

      virtual CameraModelConcept* copy() const = 0;

      // virtual auto distort(const Eigen::Vector2f&) const -> Eigen::Vector2f = 0;
      // virtual auto undistort(const Eigen::Vector2f&) const
      //     -> Eigen::Vector2f = 0;

      virtual auto project(const Eigen::Vector3f&) const -> Eigen::Vector2f = 0;

      virtual auto backproject(const Eigen::Vector2f&) const
          -> Eigen::Vector3f = 0;
    };

    template <typename Impl>
    struct CameraModelImpl : CameraModelConcept
    {
      CameraModelImpl(Impl impl)
        : _impl(std::move(impl))
      {
      }

      CameraModelConcept* copy() const
      {
        return new CameraModelImpl{*this};
      }

      virtual auto project(const Eigen::Vector3f& x) const -> Eigen::Vector2f
      {
        return _impl.project(x);
      }

      virtual auto backproject(const Eigen::Vector2f& x) const
          -> Eigen::Vector3f
      {
        return _impl.backproject(x);
      }

      Impl _impl;
    };

    std::unique_ptr<CameraModelConcept> _self;
  };

}  // namespace DO::Sara
