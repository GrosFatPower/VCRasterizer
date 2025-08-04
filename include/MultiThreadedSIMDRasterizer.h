#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <immintrin.h>  // AVX/SSE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define SIMD_ALIGN alignas(32)

struct Triangle {
  glm::vec3 vertices[3];
  uint32_t color;
  uint32_t materialId;

  Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, uint32_t col = 0xFF0000FF)
    : color(col), materialId(0) {
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;
  }
};

struct TransformedTriangle {
  glm::vec4 screenVertices[3];
  uint32_t color;
  float area;
  bool valid;

  // Edge function coefficients pour SIMD
  SIMD_ALIGN float edgeA[3];
  SIMD_ALIGN float edgeB[3];
  SIMD_ALIGN float edgeC[3];
};

struct Tile {
  int x, y;
  int width, height;
  std::vector<const TransformedTriangle*> triangles;
};

class MultiThreadedSIMDRasterizer
{
public:
  MultiThreadedSIMDRasterizer(int w, int h, int numThreads = 0);

  ~MultiThreadedSIMDRasterizer();

  void Clear(uint32_t color = 0x000000FF);

  int InitSingleTriangleScene();
  int InitMultipleTrianglesScene(const int nbTris = 100);

  // Transformation et culling des triangles en batch
  void TransformTriangles(const std::vector<Triangle>& triangles, const glm::mat4& mvp, std::vector<TransformedTriangle>& oTransformed);

  // Binning des triangles par tuile (avec overlap detection)
  void BinTrianglesToTiles(const std::vector<TransformedTriangle>& triangles);

  // Worker thread function
  void WorkerThreadFunction();

  // Rendu d'une tuile avec SIMD
  void RenderTile(const Tile & tile);

  void RenderTriangleInTile(const TransformedTriangle& tri, const Tile & tile);

  // Test SIMD de 8 pixels
  __m256i TestPixels8x(float startX, float y, const TransformedTriangle& tri);

  void InterpolateDepth8x(float startX, float y, const TransformedTriangle& tri,
    const glm::vec3& depths, const glm::vec3& wValues, float* output);

  void SetupEdgeFunctions(TransformedTriangle& tri);

  glm::vec4 TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp);

  // Fonction principale de rendu
  void RenderTriangles(const std::vector<Triangle>& triangles, const glm::mat4& mvp);

  void RenderRotatingScene(float time);

  // Mise à jour des statistiques FPS
  void UpdateFrameStats();
  double GetCurrentFPS() const;
  double GetAverageFrameTime() const;
  double GetLastFrameTime() const;
  int GetFrameCount() const;
  double GetTotalRenderTime() const;
  void PrintPerformanceStats() const;

  const uint32_t* GetColorBuffer() const { return _ColorBuffer.data(); }
  int GetWidth() const { return _ScreenWidth; }
  int GetHeight() const { return _ScreenHeight; }

  void SetBackfaceCullingEnabled(bool enabled) { _BackfaceCullingEnabled = enabled; }

private:
  int _ScreenWidth, _ScreenHeight;
  int _TileCountX, _TileCountY;

  std::vector<uint32_t> _ColorBuffer;
  std::vector<float> _DepthBuffer;
  std::vector<Tile> _Tiles;

  // Thread pool
  std::vector<std::thread> _WorkerThreads;
  std::atomic<int> A_NextTileIndex{ 0 };
  std::atomic<bool> A_RenderingActive{ false };
  std::condition_variable _RenderCV;
  std::mutex _RenderMutex;
  std::condition_variable _TilesDoneCV;
  std::mutex _TilesDoneMutex;

  // Scene
  std::vector<Triangle> _Triangles;
  std::vector<TransformedTriangle> _Transformed;

  // Stats de performance
  std::atomic<int> A_TrianglesProcessed{ 0 };
  std::atomic<int> A_PixelsRendered{ 0 };

  // Compteurs FPS et timing
  mutable std::mutex _StatsMutex;
  std::chrono::high_resolution_clock::time_point _LastFrameTime;
  std::chrono::high_resolution_clock::time_point _StartTime;
  std::vector<double> _FrameTimes;
  double _CurrentFPS = 0.0;
  double _AvgFrameTime = 0.0;
  int _FrameCount = 0;
  static constexpr int FPS_HISTORY_SIZE = 60; // Moyenner sur 60 frames

  bool _BackfaceCullingEnabled = true;

  static constexpr int TILE_SIZE = 64;
};

// Classe de benchmark pour comparer single-thread vs multi-thread
class MTRasterizerBenchmark
{
public:
  static void ComparePerformance(int triangleCount = 1000);
};
