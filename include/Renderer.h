#pragma once

#include "DatatTypes.h"
#include <vector>

class Renderer
{
public:
  Renderer(int w, int h) : _ScreenWidth(w), _ScreenHeight(h)
  {
    _ColorBuffer.resize(_ScreenWidth * _ScreenHeight);
    _DepthBuffer.resize(_ScreenWidth * _ScreenHeight);
  }

  virtual ~Renderer() = default;

  virtual int InitScene(const int nbTris = 100)
  {
    _Triangles.clear();

    LoadTriangles(_Triangles, nbTris);

    return 0;
  }

  virtual void RenderRotatingScene(float time) = 0;

  // Accès au buffer pour affichage
  const uint32_t* GetColorBuffer() const { return _ColorBuffer.data(); }
  const float* GetDepthBuffer() const { return _DepthBuffer.data(); }
  int GetWidth() const { return _ScreenWidth; }
  int GetHeight() const { return _ScreenHeight; }

  const std::vector<Triangle>& GetTriangles() const { return _Triangles; }
  virtual void SetTriangles(const std::vector<Triangle>& triangles) { _Triangles = triangles; }

  bool GetBackfaceCullingEnabled() const { return _EnableBackfaceCulling; }
  bool GetEnableSIMD() const { return _EnableSIMD; }
  void SetBackfaceCullingEnabled(bool enabled) { _EnableBackfaceCulling = enabled; }
  void SetEnableSIMD(bool enabled) { _EnableSIMD = enabled; }

protected:
  int _ScreenWidth, _ScreenHeight;

  std::vector<uint32_t> _ColorBuffer;
  std::vector<float> _DepthBuffer;

  // Scene
  std::vector<Triangle> _Triangles;

  bool _EnableBackfaceCulling = true;
  bool _EnableSIMD = true;
};
