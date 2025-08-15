#pragma once

// Configuration des warnings pour différents compilateurs
#ifdef _MSC_VER
#pragma warning(disable: 4324) // Disable structure padding warning
#elif defined(__clang__) || defined(__GNUC__)
// Pour Clang/GCC, les warnings de padding sont moins fréquents
#pragma GCC diagnostic ignored "-Wpadded"
#endif

#include "DatatTypes.h"
#include "Renderer.h"
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <immintrin.h>  // AVX/SSE
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
  void RenderTile(const Tile & tile);

  void RenderTriangleInTile(const TransformedTriangle& tri, const Tile & tile);
  void RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile);

  bool TestPixels1x(float x, float y, const TransformedTriangle& tri);
  __m256i TestPixels8x(float startX, float y, const TransformedTriangle& tri); // Test SIMD de 8 pixels

  float InterpolateDepth1x_InverseZ(float x, float y, const TransformedTriangle& tri);
  void InterpolateDepth8x(float startX, float y, const TransformedTriangle& tri, const glm::vec3& depths, const glm::vec3& wValues, float* output);
  void InterpolateDepth8x_InverseZ(float startX, float y, const TransformedTriangle& tri, float* output);

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
