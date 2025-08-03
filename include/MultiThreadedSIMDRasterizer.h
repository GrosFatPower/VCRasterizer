#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <future>
#include <chrono>
#include <immintrin.h>  // AVX/SSE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define SIMD_ALIGN alignas(32)
#define TILE_SIZE 32

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

struct RenderTile {
  int x, y;
  int width, height;
  std::vector<const TransformedTriangle*> triangles;
};

class MultiThreadedSIMDRasterizer
{
private:
  int screenWidth, screenHeight;
  int tileCountX, tileCountY;

  std::vector<uint32_t> colorBuffer;
  std::vector<float> depthBuffer;
  std::vector<RenderTile> tiles;

  // Thread pool
  std::vector<std::thread> workerThreads;
  std::atomic<int> nextTileIndex{ 0 };
  std::atomic<bool> renderingActive{ false };

  // Stats de performance
  //std::atomic<int> trianglesProcessed{ 0 };
  //std::atomic<int> pixelsRendered{ 0 };

  // Compteurs FPS et timing
  mutable std::mutex statsMutex;
  std::chrono::high_resolution_clock::time_point lastFrameTime;
  std::chrono::high_resolution_clock::time_point startTime;
  std::vector<double> frameTimes;
  double currentFPS = 0.0;
  double avgFrameTime = 0.0;
  int frameCount = 0;

  static constexpr int FPS_HISTORY_SIZE = 60; // Moyenner sur 60 frames

public:
  MultiThreadedSIMDRasterizer(int w, int h, int numThreads = 0);

  ~MultiThreadedSIMDRasterizer();

  void clear(uint32_t color = 0x000000FF);

  // Transformation et culling des triangles en batch
  std::vector<TransformedTriangle> transformTriangles(const std::vector<Triangle>& triangles, const glm::mat4& mvp);

  // Binning des triangles par tuile (avec overlap detection)
  void binTrianglesToTiles(const std::vector<TransformedTriangle>& triangles);

  // Worker thread function
  void workerThreadFunction();

  // Rendu d'une tuile avec SIMD
  void renderTile(const RenderTile& tile);

  void renderTriangleInTile(const TransformedTriangle& tri, const RenderTile& tile);

  // Test SIMD de 8 pixels
  __m256i testPixels8x(float startX, float y, const TransformedTriangle& tri);

  void interpolateDepth8x(float startX, float y, const TransformedTriangle& tri,
    const glm::vec3& depths, const glm::vec3& wValues, float* output);

  void setupEdgeFunctions(TransformedTriangle& tri);

  glm::vec4 transformVertex(const glm::vec3& vertex, const glm::mat4& mvp);

  // Fonction principale de rendu
  void renderTriangles(const std::vector<Triangle>& triangles, const glm::mat4& mvp);

  void renderRotatingTriangle(float time);

  void renderRotatingTriangles(float time);

  // Mise à jour des statistiques FPS
  void updateFrameStats();

  // Getters thread-safe pour les stats
  double getCurrentFPS() const;

  double getAverageFrameTime() const;

  double getLastFrameTime() const;

  int getFrameCount() const;

  double getTotalRenderTime() const;

  // Affichage des stats de performance détaillées
  void printPerformanceStats() const;

  const uint32_t* getColorBuffer() const { return colorBuffer.data(); }
  int getWidth() const { return screenWidth; }
  int getHeight() const { return screenHeight; }
};

// Classe de benchmark pour comparer single-thread vs multi-thread
class MTRasterizerBenchmark
{
public:
  static void comparePerformance(int triangleCount = 1000);
};
