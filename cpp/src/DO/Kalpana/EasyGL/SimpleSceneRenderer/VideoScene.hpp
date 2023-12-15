#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Kalpana/EasyGL/Objects/Scene.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedImage.hpp>
#include <DO/Kalpana/EasyGL/Objects/TexturedQuad.hpp>
#include <DO/Kalpana/EasyGL/Renderer/TextureRenderer.hpp>


namespace DO::Kalpana::GL {

  struct VideoScene : BasicScene
  {
    auto set_viewport_box(const AxisAlignedBoundingBox<int>& box) -> void
    {
      BasicScene::_viewport = Viewport{box};
      update_projection(_scale);
    }

    auto init(const ImageView<Bgr8>& image) -> void
    {
      // Initialize the OpenGL device texture.
      _texture.initialize(image, 0);

      // Initialize the quad vertices.
      const auto aspect_ratio =
          static_cast<float>(image.width()) / image.height();
      auto vertices = _texture_quad.host_vertices().matrix();
      vertices.col(0) *= aspect_ratio;
      _texture_quad.initialize();

      // Init the texture renderer.
      _texture_renderer.initialize();

      // Projection-model-view matrices.
      _model_view.setIdentity();
      _projection = _viewport.orthographic_projection(_scale);
    }

    auto deinit() -> void
    {
      _texture.destroy();
      _texture_quad.destroy();
      _texture_renderer.destroy();
    }

    auto update_texture(const ImageView<Bgr8>& image) -> void
    {
      _texture.reset(image);
    }

    auto render() -> void
    {
      glViewport(_viewport.top_left().x(), _viewport.top_left().y(),
                 _viewport.width(), _viewport.height());
      _texture_renderer.render(_texture, _texture_quad, _model_view,
                               _projection);
    }

    auto update_projection(const float scale) -> void
    {
      _scale = scale;
      _projection = _viewport.orthographic_projection(_scale);
    }

    auto scale() const -> float
    {
      return _scale;
    }

    TexturedImage2D _texture;
    TexturedQuad _texture_quad;
    TextureRenderer _texture_renderer;
    float _scale = 0.5f;
  };

}  // namespace DO::Kalpana::GL
