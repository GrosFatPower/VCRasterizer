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
  virtual ~MultiThreadedSIMDRasterizer();

  virtual int InitScene(const int nbTris = 100);

  virtual void RenderRotatingScene(float time);

  virtual void SetTriangles(const std::vector<Triangle> & triangles);

protected:

  void Clear(uint32_t color = 0xADD8E6FF);

  // Binning des triangles par tuile (avec overlap detection)
  void BinTrianglesToTiles();

  // Worker thread function
  void WorkerThreadFunction();

  // Rendu d'une tuile avec SIMD
  void RenderTile(const Tile& tile);

  void RenderTriangles(const glm::mat4& mvp);
  void RenderTrianglesInBatch(const glm::mat4& mvp);

  void SetupTriangleData(TransformedTriangle& tri);

  glm::vec4 TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp);

  void RenderTriangleInTile1x(const TransformedTriangle& tri, const Tile& tile);
  void RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile);
  void RenderTriangleInTile4x(const TransformedTriangle& tri, const Tile& tile);

  bool TestPixels1x(float x, float y, const TransformedTriangle& tri);
#ifdef SIMD_ARM_NEON
  uint32x4_t TestPixels4x(float startX, float y, const TransformedTriangle& tri); // ARM NEON
#endif
#ifdef SIMD_AVX2
  __m256i TestPixels8x(float startX, float y, const TransformedTriangle& tri); // AVX2
#endif

  float InterpolateDepth1x(float x, float y, const TransformedTriangle& tri);
#ifdef SIMD_ARM_NEON
  void InterpolateDepth4x(float startX, float y, const TransformedTriangle& tri, float* output); // ARM NEON
#endif
#ifdef SIMD_AVX2
  void InterpolateDepth8x(float startX, float y, const TransformedTriangle& tri, float* output); // AVX2
#endif

private:
  int _TileCountX, _TileCountY;
  std::vector<Tile> _Tiles;

  // Scene
  std::vector<TransformedTriangle> _Transformed;

  // Thread pool
  std::vector<std::thread> _WorkerThreads;
  std::atomic<int>         _NextTileIndexAtomic{ 0 };
  std::atomic<bool>        _RenderingActiveAtomic{ false };
  std::atomic<bool>        _TerminateThreadWorkAtomic{ false };
  std::condition_variable  _RenderCV;
  std::mutex               _RenderMutex;
  std::condition_variable  _TilesDoneCV;
  std::mutex               _TilesDoneMutex;

  static constexpr int TILE_SIZE = 32;
};
