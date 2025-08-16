#pragma once

#include "SIMDUtils.h"
#include "DataTypes.h"
#include "Renderer.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>



class MultiThreadedSIMDRasterizer : public Renderer
{
public:
  MultiThreadedSIMDRasterizer(int w, int h, int numThreads = 0);

  ~MultiThreadedSIMDRasterizer();

  virtual int InitScene(const int nbTris = 100);

  virtual void RenderRotatingScene(float time);

protected:

  void Clear(uint32_t color = 0x000000FF);

  // Transformation et culling des triangles en batch
  void TransformTriangles(const glm::mat4& mvp);

  // Binning des triangles par tuile (avec overlap detection)
  void BinTrianglesToTiles();

  // Worker thread function
  void WorkerThreadFunction();

  // Rendu d'une tuile avec SIMD
  void RenderTile(const Tile& tile);

  void RenderTriangleInTile(const TransformedTriangle& tri, const Tile& tile);
  void RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile);
  void RenderTriangleInTile4x(const TransformedTriangle& tri, const Tile& tile); // Pour ARM NEON

  bool TestPixels1x(float x, float y, const TransformedTriangle& tri);

  // Test SIMD de pixels - adapte selon la plateforme
#ifdef SIMD_ARM_NEON
  uint32x4_t TestPixels4x_NEON(float startX, float y, const TransformedTriangle& tri);
#endif
#ifdef SIMD_AVX2
  __m256i TestPixels8x_AVX2(float startX, float y, const TransformedTriangle& tri);
#endif

  float InterpolateDepth1x_InverseZ(float x, float y, const TransformedTriangle& tri);

  // Interpolation de profondeur - adapte selon la plateforme
#ifdef SIMD_ARM_NEON
  void InterpolateDepth4x_InverseZ_NEON(float startX, float y, const TransformedTriangle& tri, float* output);
#endif
#ifdef SIMD_AVX2
  void InterpolateDepth8x_InverseZ_AVX2(float startX, float y, const TransformedTriangle& tri, float* output);
#endif

  void SetupTriangleData(TransformedTriangle& tri);

  glm::vec4 TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp);

  // Fonction principale de rendu
  void RenderTriangles(const glm::mat4& mvp);
  void RenderTrianglesInBatch(const glm::mat4& mvp);

  virtual void SetTriangles(const std::vector<Triangle>& triangles);

private:
  int _TileCountX, _TileCountY;
  std::vector<Tile> _Tiles;

  // Thread pool
  std::vector<std::thread> _WorkerThreads;
  std::atomic<int> A_NextTileIndex{ 0 };
  std::atomic<bool> A_RenderingActive{ false };
  std::atomic<bool> A_ThreadsShouldRun{ true };
  std::condition_variable _RenderCV;
  std::mutex _RenderMutex;
  std::condition_variable _TilesDoneCV;
  std::mutex _TilesDoneMutex;

  // Scene
  std::vector<TransformedTriangle> _Transformed;

  static constexpr int TILE_SIZE = 32;
};
