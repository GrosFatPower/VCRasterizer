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
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <condition_variable>
#include <memory>
#include <immintrin.h>  // AVX/SSE
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

  OptimizedMultiThreadedSIMDRasterizer(int w, int h, int numThreads = 0);
  virtual ~OptimizedMultiThreadedSIMDRasterizer();

  // Interface principale
  virtual int InitScene(const int nbTris = 100) override;
  virtual void RenderRotatingScene(float time) override;
  virtual void SetTriangles(const std::vector<Triangle>& triangles) override;

  // Interface Renderer
  virtual void SetEnableSIMD(bool enable);
  virtual bool GetEnableSIMD() const;
  virtual void SetBackfaceCullingEnabled(bool enable);
  virtual bool GetBackfaceCullingEnabled() const;

  // Méthodes de configuration et debugging
  void SetRenderMode(RenderMode mode);
  void PrintPerformanceStats() const;
  void ResetPerformanceCounters();
  double BenchmarkRenderTime(int frames = 100);
  bool ValidateResults(const OptimizedMultiThreadedSIMDRasterizer& reference);

protected:

  // Configuration
  static constexpr int TILE_SIZE = 64;
  static constexpr int MIN_TRIANGLES_PER_TILE = 4;

  // Structures de données cache-friendly
  struct alignas(64) ThreadLocalData
  {
    alignas(32) float depthBuffer[TILE_SIZE * TILE_SIZE];
    alignas(32) uint32_t colorBuffer[TILE_SIZE * TILE_SIZE];
    std::vector<const TransformedTriangle*> localTriangles;

    ThreadLocalData()
    {
      // Initialize buffers to default values
      std::fill_n(depthBuffer, TILE_SIZE * TILE_SIZE, std::numeric_limits<float>::infinity());
      std::fill_n(colorBuffer, TILE_SIZE * TILE_SIZE, 0);
      localTriangles.reserve(1000);
    }
  };

  struct alignas(64) TileData
  {
    Tile tile;
    std::vector<const TransformedTriangle*> triangles;
    std::atomic<int> triangleCount;
    std::atomic<bool> needsProcessing;

    TileData()
      : triangleCount(0)
      , needsProcessing(false)
    {
      triangles.reserve(500);
    }

    // Delete copy constructor and assignment operator due to atomic members
    TileData(const TileData&) = delete;
    TileData& operator=(const TileData&) = delete;

    // Add move constructor and assignment operator
    TileData(TileData&& other) noexcept
      : tile(std::move(other.tile))
      , triangles(std::move(other.triangles))
      , triangleCount(other.triangleCount.load())
      , needsProcessing(other.needsProcessing.load())
    {
    }

    TileData& operator=(TileData&& other) noexcept {
      if (this != &other) {
        tile = std::move(other.tile);
        triangles = std::move(other.triangles);
        triangleCount = other.triangleCount.load();
        needsProcessing = other.needsProcessing.load();
      }
      return *this;
    }
  };

protected:
  // Pipeline de rendu optimisé
  void Clear(uint32_t color = 0x000000FF);
  void TransformTrianglesVectorized(const glm::mat4& mvp);
  void HierarchicalBinning();
  void RenderTrianglesMultiThreaded();

  // Threading optimisé avec work stealing
  void WorkerThreadFunctionOptimized(int threadId);
  bool StealWork(int threadId, int& outTileIndex);

  // Rendu de tuiles avec SIMD
  void RenderTileAVX2(const Tile& tile, ThreadLocalData* localData);
  void RenderTriangleInTile16x(const TransformedTriangle& tri, const Tile& tile, ThreadLocalData* localData);
  void RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile);

  // Tests de pixels optimisés
  __m256i TestPixels8xOptimized(float startX, float y, const TransformedTriangle& tri);
  bool TestPixels1x(float x, float y, const TransformedTriangle& tri);
  bool TestBlockVisibility(int blockX, int blockY, int blockW, int blockH, const TransformedTriangle& tri);

  // Interpolation de profondeur
  void InterpolateDepth8x_InverseZ(float startX, float y, const TransformedTriangle& tri, float* output);
  float InterpolateDepth1x_InverseZ(float x, float y, const TransformedTriangle& tri);

  // Mise à jour des buffers
  void UpdateZBuffer8x(int pixelIndex, const float* depths, const __m256i& mask, uint32_t color);
  void UpdateLocalBuffer8x(int localX, int localY, int tileWidth, const float* depths,
    const __m256i& mask, uint32_t color, ThreadLocalData* localData);
  void CopyTileToMainBuffer(const Tile& tile, ThreadLocalData* localData);

  // Utilitaires de transformation SIMD
  glm::vec4 TransformVertexSIMD(const glm::vec3& vertex, const __m256& mvpRow0,
    const __m256& mvpRow1, const __m256& mvpRow2, const __m256& mvpRow3);
  void SetupTriangleDataOptimized(TransformedTriangle& tri);
  void InitializeLookupTables();

private:
  // Données de tiling
  int _TileCountX, _TileCountY;
  std::vector<TileData> _OptimizedTiles;

  // Thread pool optimisé
  int _NumThreads;
  std::vector<std::thread> _WorkerThreads;
  std::vector<std::unique_ptr<ThreadLocalData>> _ThreadLocalData;

  // Synchronisation avec work stealing
  std::atomic<int> _WorkStealingIndex{ 0 };
  //std::vector<std::atomic<int>> _ThreadWorkIndices;
  std::deque<std::atomic<int>> _ThreadWorkIndices;
  std::atomic<bool> _RenderingActive{ false };
  std::atomic<bool> _ThreadsShouldRun{ true };
  std::condition_variable _RenderCV;
  std::mutex _RenderMutex;
  std::condition_variable _TilesDoneCV;
  std::mutex _TilesDoneMutex;

  // Scene et triangles transformés
  std::vector<Triangle> _Triangles;
  std::vector<TransformedTriangle> _Transformed;

  // Configuration et flags
  RenderMode _CurrentRenderMode = RenderMode::AVX2;
  bool _EnableSIMD = true;
  bool _BackfaceCullingEnabled = true;

  // Lookup tables pour optimisations
  alignas(32) float _EdgeLUT[256];

  // Compteurs de performance
  std::atomic<uint64_t> _TrianglesProcessed{ 0 };
  std::atomic<uint64_t> _PixelsRasterized{ 0 };

  // Statistiques de profiling
  struct PerformanceStats {
    std::atomic<uint64_t> transformTime{ 0 };
    std::atomic<uint64_t> binningTime{ 0 };
    std::atomic<uint64_t> renderTime{ 0 };
    std::atomic<uint64_t> frameCount{ 0 };
  } _PerfStats;
};

// Classe de benchmark pour comparaisons
class RasterizerBenchmark {
public:
  struct BenchmarkResult {
    std::string name;
    double avgFrameTimeMs;
    double fps;
    uint64_t trianglesPerSecond;
    double efficiency; // triangles par seconde par thread
  };

  static std::vector<BenchmarkResult> CompareImplementations(
    const std::vector<Triangle>& triangles,
    int width, int height,
    int frames = 100);

  static void PrintBenchmarkResults(const std::vector<BenchmarkResult>& results);

  // Benchmark spécifique SIMD
  static void BenchmarkSIMDModes(OptimizedMultiThreadedSIMDRasterizer& rasterizer,
    const std::vector<Triangle>& triangles,
    int frames = 100);
};

// Profiler pour analyse détaillée des performances
class RasterizerProfiler {
public:
  struct ProfileData {
    uint64_t totalTimeUs = 0;
    uint64_t callCount = 0;
    uint64_t minTimeUs = UINT64_MAX;
    uint64_t maxTimeUs = 0;

    double GetAverageMs() const {
      return callCount > 0 ? (double)totalTimeUs / (callCount * 1000.0) : 0.0;
    }
  };

  class ScopedTimer {
  public:
    ScopedTimer(RasterizerProfiler& profiler, const std::string& name);
    ~ScopedTimer();

  private:
    RasterizerProfiler& _profiler;
    std::string _name;
    std::chrono::high_resolution_clock::time_point _start;
  };

  void AddSample(const std::string& name, uint64_t timeUs);
  void PrintReport() const;
  void Reset();

  // Macro pour simplifier l'utilisation
#define PROFILE_SCOPE(profiler, name) \
        RasterizerProfiler::ScopedTimer _timer(profiler, name)

private:
  std::unordered_map<std::string, ProfileData> _profiles;
  mutable std::mutex _profileMutex;
};

// Utilitaires SIMD pour tests et validation
namespace SIMDUtils {
  // Test de support des instructions SIMD
  inline bool HasAVX2Support()
  {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 5)) != 0; // AVX2 support
  }

  inline bool HasAVX512Support()
  {
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    return (cpuInfo[1] & (1 << 16)) != 0; // AVX512F support
  }

  // Benchmark des différentes largeurs SIMD
  void BenchmarkSIMDWidths();

  // Validation de cohérence des calculs SIMD vs scalaire
  bool ValidateSIMDAccuracy(const std::vector<float>& input,
    float(*scalarFunc)(float),
    void(*simdFunc)(const float*, float*, size_t));
}
