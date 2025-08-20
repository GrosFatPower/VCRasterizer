#pragma once

#include "SIMDUtils.h"
#include "DataTypes.h"
#include "Renderer.h"
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <condition_variable>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class OptimizedMultiThreadedSIMDRasterizer : public Renderer
{
public:
  enum class RenderMode
  {
    SCALAR,
    SSE,
    AVX2,
    AVX512
  };

public:
  OptimizedMultiThreadedSIMDRasterizer(int w, int h, int numThreads = 0);
  virtual ~OptimizedMultiThreadedSIMDRasterizer();

  // Interface principale
  virtual int InitScene(const int nbTris = 100) override;
  virtual void RenderRotatingScene(float time) override;
  virtual void SetTriangles(const std::vector<Triangle>& triangles) override;

  // Methodes de configuration et debugging
  void SetRenderMode(RenderMode mode);

protected:

  // Configuration
  static constexpr int TILE_SIZE = 64;
  static constexpr int MIN_TRIANGLES_PER_TILE = 4;

  // Structures de donnees cache-friendly
  struct alignas(64) ThreadLocalData
  {
    alignas(32) float                       _DepthBuffer[TILE_SIZE * TILE_SIZE];
    alignas(32) uint32_t                    _ColorBuffer[TILE_SIZE * TILE_SIZE];
    ThreadLocalData();
  };

  struct alignas(64) TileData
  {
    Tile                                    _Tile;
    std::atomic<int>                        _TriangleCountAtomic;
    std::atomic<bool>                       _NeedsProcessingAtomic;

    TileData();

    // Add move constructor and assignment operator
    TileData(TileData&& other) noexcept;
    TileData& operator=(TileData&& other) noexcept;

    // Delete copy constructor and assignment operator due to atomic members
    TileData(const TileData&) = delete;
    TileData& operator=(const TileData&) = delete;
  };

protected:
  // Pipeline de rendu optimise
  void Clear(uint32_t color = G_DEFAULT_COLOR);
  void HierarchicalBinning();
  void RenderTrianglesMultiThreaded();

  // Threading optimise avec work stealing
  void WorkerThreadFunctionOptimized(int threadId);
  //bool StealWork(int threadId, int& outTileIndex);

  glm::vec4 TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp);
  void TransformTriangles(const glm::mat4& mvp);

  void RenderTile(const Tile& tile, ThreadLocalData* localData);

  bool TestBlockVisibility(int blockX, int blockY, int blockW, int blockH, const TransformedTriangle& tri);

  bool TestPixels1x(float x, float y, const TransformedTriangle& tri);
  float InterpolateDepth1x(float x, float y, const TransformedTriangle& tri);

  void RenderTriangleInTile(const TransformedTriangle& tri, const Tile& tile, ThreadLocalData* localData);

  void CopyTileToMainBuffer(const Tile& tile, ThreadLocalData* localData);

  void SetupTriangleData(TransformedTriangle& tri);
  void InitializeLookupTables();

#ifdef SIMD_AVX2
  void Clear8x(uint32_t color = G_DEFAULT_COLOR);

  void TransformTrianglesAVX2(const glm::mat4& mvp);

  void RenderTileAVX2(const Tile& tile, ThreadLocalData* localData);
  
  void RenderTriangleInTile16x(const TransformedTriangle& tri, const Tile& tile, ThreadLocalData* localData);
  void RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile);
  
  __m256i TestPixels8x(float startX, float y, const TransformedTriangle& tri);
  void InterpolateDepth8x(float startX, float y, const TransformedTriangle& tri, float* output);
  
  void UpdateZBuffer8x(int pixelIndex, const float* depths, const __m256i& mask, uint32_t color);
  void UpdateLocalBuffer8x(int localX, int localY, int tileWidth, const float* depths, const __m256i& mask, uint32_t color, ThreadLocalData* localData);

  void CopyTileToMainBuffer8x(const Tile& tile, ThreadLocalData* localData);
  
  glm::vec4 TransformVertexAVX2(const glm::vec3& vertex, const __m256& mvpRow0, const __m256& mvpRow1, const __m256& mvpRow2, const __m256& mvpRow3);
#endif

private:
  // Donnees de tiling
  int _TileCountX, _TileCountY;
  std::vector<TileData> _OptimizedTiles;

  // Thread pool optimise
  int                                           _NumThreads;
  std::vector<std::thread>                      _WorkerThreads;
  std::vector<std::unique_ptr<ThreadLocalData>> _ThreadLocalData;

  // Synchronisation avec work stealing
  std::atomic<int>                _WorkStealingIndexAtomic{ 0 };
  //std::vector<std::atomic<int>> _ThreadWorkIndices;
  std::deque<std::atomic<int>>    _ThreadWorkIndices;
  std::atomic<bool>               _RenderingActiveAtomic{ false };
  std::atomic<bool>               _ThreadsShouldRunAtomic{ true };
  std::condition_variable         _RenderCV;
  std::mutex                      _RenderMutex;
  std::condition_variable         _TilesDoneCV;
  std::mutex                      _TilesDoneMutex;

  // Scene
  std::vector<TransformedTriangle> _Transformed;
};

inline OptimizedMultiThreadedSIMDRasterizer::ThreadLocalData::ThreadLocalData()
{
  // Initialize buffers to default values
  std::fill_n(_DepthBuffer, TILE_SIZE * TILE_SIZE, std::numeric_limits<float>::infinity());
  std::fill_n(_ColorBuffer, TILE_SIZE * TILE_SIZE, 0);
}

inline OptimizedMultiThreadedSIMDRasterizer::TileData::TileData()
: _TriangleCountAtomic(0)
, _NeedsProcessingAtomic(false)
{
}

// Add move constructor and assignment operator
inline OptimizedMultiThreadedSIMDRasterizer::TileData::TileData(TileData&& other) noexcept
  : _Tile(std::move(other._Tile))
  , _TriangleCountAtomic(other._TriangleCountAtomic.load())
  , _NeedsProcessingAtomic(other._NeedsProcessingAtomic.load())
{
}

inline OptimizedMultiThreadedSIMDRasterizer::TileData& OptimizedMultiThreadedSIMDRasterizer::TileData::operator=(TileData&& other) noexcept
{
  if (this != &other) {
    _Tile = std::move(other._Tile);
    _TriangleCountAtomic = other._TriangleCountAtomic.load();
    _NeedsProcessingAtomic = other._NeedsProcessingAtomic.load();
  }
  return *this;
}
