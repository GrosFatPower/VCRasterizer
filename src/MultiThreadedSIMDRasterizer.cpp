#include "MultiThreadedSIMDRasterizer.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <future>
#include <glm/gtc/matrix_transform.hpp>

MultiThreadedSIMDRasterizer::MultiThreadedSIMDRasterizer(int w, int h, int numThreads)
  : Renderer(w, h)
{
  // Calcul du nombre de tuiles
  _TileCountX = (_ScreenWidth + TILE_SIZE - 1) / TILE_SIZE;
  _TileCountY = (_ScreenHeight + TILE_SIZE - 1) / TILE_SIZE;

  // Initialisation des tuiles
  _Tiles.reserve(_TileCountX * _TileCountY);
  for (int ty = 0; ty < _TileCountY; ++ty)
  {
    for (int tx = 0; tx < _TileCountX; ++tx)
    {
      Tile tile;
      tile.x = tx * TILE_SIZE;
      tile.y = ty * TILE_SIZE;
      tile.width = std::min(TILE_SIZE, _ScreenWidth - tile.x);
      tile.height = std::min(TILE_SIZE, _ScreenHeight - tile.y);
      _Tiles.push_back(tile);
    }
  }

  // Initialisation du thread pool
  int threadCount = (numThreads == 0) ? std::thread::hardware_concurrency() : numThreads;
  threadCount = std::min(threadCount, (int)_Tiles.size()); // Pas plus de threads que de tuiles

  for (int i = 0; i < threadCount; ++i)
    _WorkerThreads.emplace_back(&MultiThreadedSIMDRasterizer::WorkerThreadFunction, this);
}

MultiThreadedSIMDRasterizer::~MultiThreadedSIMDRasterizer()
{
  {
    std::unique_lock<std::mutex> lock(_RenderMutex);
    A_ThreadsShouldRun = false; // Signal threads to exit
    A_RenderingActive = false;
  }
  _RenderCV.notify_all(); // Wake up all threads

  // Wait for all threads to finish
  for (auto& thread : _WorkerThreads)
  {
    if (thread.joinable())
      thread.join();
  }
}

void MultiThreadedSIMDRasterizer::Clear(uint32_t color)
{
  // Clear en parallele par chunks
  const int chunkSize = 64000; // ~256KB chunks
  int numChunks = ((int)_ColorBuffer.size() + chunkSize - 1) / chunkSize;

  std::vector<std::future<void>> futures;
  for (int i = 0; i < numChunks; ++i)
  {
    int start = i * chunkSize;
    int end = std::min(start + chunkSize, (int)_ColorBuffer.size());

    futures.push_back(
      std::async(
        std::launch::async, [this, start, end, color]()
        {
          std::fill(_ColorBuffer.begin() + start, _ColorBuffer.begin() + end, color);
          std::fill(_DepthBuffer.begin() + start, _DepthBuffer.begin() + end, std::numeric_limits<float>::max());
        }
      )
    );
  }

  // Attendre la fin du Clear
  for (auto& future : futures)
    future.wait();
}

// Transformation et culling des triangles en batch
void MultiThreadedSIMDRasterizer::TransformTriangles(const glm::mat4& mvp)
{
  // Traitement par batch pour optimiser le cache
  const int batchSize = 64;
  for (size_t i = 0; i < _Triangles.size(); i += batchSize)
  {
    size_t end = std::min(i + batchSize, _Triangles.size());

    for (size_t j = i; j < end; ++j)
    {
      TransformedTriangle& tri = _Transformed[j];
      tri.color = _Triangles[j].color;
      tri.valid = false;

      // Transformation des vertices
      for (int v = 0; v < 3; ++v)
        tri.screenVertices[v] = TransformVertex(_Triangles[j].vertices[v], mvp);

      // Backface culling
      glm::vec2 edge1 = glm::vec2(tri.screenVertices[1].x - tri.screenVertices[0].x,
        tri.screenVertices[1].y - tri.screenVertices[0].y);
      glm::vec2 edge2 = glm::vec2(tri.screenVertices[2].x - tri.screenVertices[0].x,
        tri.screenVertices[2].y - tri.screenVertices[0].y);

      float crossProduct = edge1.x * edge2.y - edge1.y * edge2.x;

      if (_EnableBackfaceCulling && crossProduct <= 0)
        continue;

      tri.area = crossProduct * 0.5f;

      // Setup edge functions
      SetupTriangleData(tri);

      // Frustum culling basique
      bool inScreen = false;
      for (int v = 0; v < 3; ++v)
      {
        if (tri.screenVertices[v].x >= 0 && tri.screenVertices[v].x < _ScreenWidth &&
          tri.screenVertices[v].y >= 0 && tri.screenVertices[v].y < _ScreenHeight)
        {
          inScreen = true;
          break;
        }
      }

      if (inScreen)
        tri.valid = true;
    }
  }
}

void MultiThreadedSIMDRasterizer::RenderTrianglesInBatch(const glm::mat4& mvp)
{
  const int batchSize = 64;
  const int numBatches = ((int)_Triangles.size() + batchSize - 1) / batchSize;

  // Creation des futures pour chaque batch
  std::vector<std::future<void>> futures;
  futures.reserve(numBatches);

  for (size_t batchStart = 0; batchStart < _Triangles.size(); batchStart += batchSize)
  {
    futures.push_back(std::async(std::launch::async, [this, batchStart, batchSize, &mvp]() {
      size_t end = std::min(batchStart + batchSize, _Triangles.size());

      for (size_t j = batchStart; j < end; ++j)
      {
        TransformedTriangle& tri = _Transformed[j];
        tri.color = _Triangles[j].color;
        tri.valid = false;

        // Transformation des vertices
        for (int v = 0; v < 3; ++v)
          tri.screenVertices[v] = TransformVertex(_Triangles[j].vertices[v], mvp);

        // Backface culling
        glm::vec2 edge1 = glm::vec2(tri.screenVertices[1].x - tri.screenVertices[0].x,
          tri.screenVertices[1].y - tri.screenVertices[0].y);
        glm::vec2 edge2 = glm::vec2(tri.screenVertices[2].x - tri.screenVertices[0].x,
          tri.screenVertices[2].y - tri.screenVertices[0].y);

        float crossProduct = edge1.x * edge2.y - edge1.y * edge2.x;

        if (_EnableBackfaceCulling && crossProduct <= 0)
          continue;

        tri.area = crossProduct * 0.5f;

        // Setup edge functions
        SetupTriangleData(tri);

        // Frustum culling basique
        bool inScreen = false;
        for (int v = 0; v < 3; ++v)
        {
          if (tri.screenVertices[v].x >= 0 && tri.screenVertices[v].x < _ScreenWidth &&
            tri.screenVertices[v].y >= 0 && tri.screenVertices[v].y < _ScreenHeight)
          {
            inScreen = true;
            break;
          }
        }

        if (inScreen)
          tri.valid = true;
      }
      }));
  }

  // Attente de la fin du traitement de tous les batches
  for (auto& future : futures)
  {
    future.wait();
  }
}

// Binning des triangles par tuile (avec overlap detection)
void MultiThreadedSIMDRasterizer::BinTrianglesToTiles()
{
  // Clear les listes de triangles des tuiles
  for (auto& tile : _Tiles)
    tile.triangles.clear();

  // Pour chaque triangle, determiner quelles tuiles il intersecte
  for (const auto& tri : _Transformed)
  {
    if (!tri.valid)
      continue;

    // Bounding box du triangle
    float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
    float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

    // Clamp aux limites de l'ecran
    minX = std::max(0.0f, minX);
    maxX = std::min((float)_ScreenWidth - 1, maxX);
    minY = std::max(0.0f, minY);
    maxY = std::min((float)_ScreenHeight - 1, maxY);

    // Calculer les tuiles intersectees
    int tileMinX = (int)minX / TILE_SIZE;
    int tileMaxX = (int)maxX / TILE_SIZE;
    int tileMinY = (int)minY / TILE_SIZE;
    int tileMaxY = (int)maxY / TILE_SIZE;

    // Ajouter le triangle aux tuiles concernees
    for (int ty = tileMinY; ty <= tileMaxY; ++ty)
    {
      for (int tx = tileMinX; tx <= tileMaxX; ++tx)
      {
        if (tx < _TileCountX && ty < _TileCountY)
        {
          int tileIndex = ty * _TileCountX + tx;
          _Tiles[tileIndex].triangles.push_back(&tri);
        }
      }
    }
  }
}

// Worker thread function
void MultiThreadedSIMDRasterizer::WorkerThreadFunction()
{
  while (true)
  {
    std::unique_lock<std::mutex> lock(_RenderMutex);
    _RenderCV.wait(lock, [this] {
      return A_RenderingActive.load() || !A_ThreadsShouldRun; // Check for exit condition
      });

    // Exit thread if shutdown is requested
    if (!A_ThreadsShouldRun)
      break;

    while (A_RenderingActive.load())
    {
      int tileIndex = A_NextTileIndex.fetch_add(1);
      if (tileIndex >= _Tiles.size())
        break;
      lock.unlock();
      RenderTile(_Tiles[tileIndex]);
      lock.lock();
    }
  }
}

// Rendu d'une tuile avec SIMD
void MultiThreadedSIMDRasterizer::RenderTile(const Tile& tile)
{
  for (const auto* tri : tile.triangles) {
    if (GetEnableSIMD()) {
#ifdef SIMD_ARM_NEON
      RenderTriangleInTile4x(*tri, tile); // ARM NEON traite 4 pixels par fois
#elif defined(SIMD_AVX2)
      RenderTriangleInTile8x(*tri, tile); // AVX2 traite 8 pixels par fois
#else
      RenderTriangleInTile(*tri, tile);   // Fallback scalaire
#endif
    }
    else {
      RenderTriangleInTile(*tri, tile);
    }
  }

  // Notify main thread if all tiles are done
  if (A_NextTileIndex.load() >= _Tiles.size()) {
    std::lock_guard<std::mutex> lock(_TilesDoneMutex);
    _TilesDoneCV.notify_one();
  }
}

void MultiThreadedSIMDRasterizer::RenderTriangleInTile(const TransformedTriangle& tri, const Tile& tile)
{
  // Bounding box du triangle dans la tuile
  float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
  float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

  // Intersection avec la tuile
  int startX = std::max(tile.x, (int)std::floor(minX));
  int endX = std::min(tile.x + tile.width - 1, (int)std::ceil(maxX));
  int startY = std::max(tile.y, (int)std::floor(minY));
  int endY = std::min(tile.y + tile.height - 1, (int)std::ceil(maxY));

  // Rasterization par blocs de pixels
  for (int y = startY; y <= endY; ++y)
  {
    for (int x = startX; x <= endX; ++x)
    {
      if (TestPixels1x(x + 0.5f, y + 0.5f, tri))
      {
        // Interpolation des profondeurs
        float interpolated_depth = InterpolateDepth1x_InverseZ(x + 0.5f, y + 0.5f, tri);

        // Z-test et ecriture des pixels
        int pixelIndex = y * _ScreenWidth + x;
        if (interpolated_depth < _DepthBuffer[pixelIndex])
        {
          _DepthBuffer[pixelIndex] = interpolated_depth;
          _ColorBuffer[pixelIndex] = tri.color;
          //A_PixelsRendered.fetch_add(1);
        }
      }
    }
  }
  //A_TrianglesProcessed.fetch_add(1);
}

#ifndef SIMD_ARM_NEON
void MultiThreadedSIMDRasterizer::RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile)
{
#ifdef SIMD_AVX2
  if (!tri.valid) return;

  float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
  float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

  int startX = std::max(tile.x, static_cast<int>(std::floor(minX)));
  int endX = std::min(tile.x + tile.width - 1, static_cast<int>(std::ceil(maxX)));
  int startY = std::max(tile.y, static_cast<int>(std::floor(minY)));
  int endY = std::min(tile.y + tile.height - 1, static_cast<int>(std::ceil(maxY)));

  // Rasterisation par blocs de 8 pixels horizontalement
  for (int y = startY; y <= endY; ++y) {
    for (int x = startX; x <= endX; x += 8) {
      // Test des 8 pixels
      __m256i inside_mask = TestPixels8x_AVX2(static_cast<float>(x) + 0.5f,
        static_cast<float>(y) + 0.5f, tri);

      // Interpolation des profondeurs pour les 8 pixels
      float depths[8];
      InterpolateDepth8x_InverseZ_AVX2(static_cast<float>(x) + 0.5f,
        static_cast<float>(y) + 0.5f, tri, depths);

      // Test de profondeur et ecriture pour chaque pixel
      for (int i = 0; i < 8 && (x + i) <= endX; ++i) {
        // Extraction du bit de masque pour le pixel i
        int mask_bit = (_mm256_movemask_ps(_mm256_castsi256_ps(inside_mask)) >> i) & 1;
        if (mask_bit) {
          int pixelX = x + i;
          int pixelIndex = y * _ScreenWidth + pixelX;

          if (depths[i] < _DepthBuffer[pixelIndex]) {
            _DepthBuffer[pixelIndex] = depths[i];
            _ColorBuffer[pixelIndex] = tri.color;
          }
        }
      }
    }
  }
#else
  // Fallback vers la version scalaire
  RenderTriangleInTile(tri, tile);
#endif
}
#endif

bool MultiThreadedSIMDRasterizer::TestPixels1x(float x, float y, const TransformedTriangle& tri)
{
  // Version scalaire qui marche sur toutes les plateformes
  for (int edge = 0; edge < 3; ++edge) {
    float edgeValue = tri.edgeA[edge] * x + tri.edgeB[edge] * y + tri.edgeC[edge];
    if (edgeValue < 0.0f) {
      return false;
    }
  }
  return true;
}

// Test SIMD de 8 pixels
//__m256i MultiThreadedSIMDRasterizer::TestPixels8x(float startX, float y, const TransformedTriangle& tri)
//{
//  __m256 x_coords = _mm256_set_ps(startX + 7, startX + 6, startX + 5, startX + 4, startX + 3, startX + 2, startX + 1, startX);
//  __m256 y_coord = _mm256_set1_ps(y);
//
//  __m256 a0 = _mm256_broadcast_ss(&tri.edgeA[0]);
//  __m256 b0 = _mm256_broadcast_ss(&tri.edgeB[0]);
//  __m256 c0 = _mm256_broadcast_ss(&tri.edgeC[0]);
//
//  __m256 a1 = _mm256_broadcast_ss(&tri.edgeA[1]);
//  __m256 b1 = _mm256_broadcast_ss(&tri.edgeB[1]);
//  __m256 c1 = _mm256_broadcast_ss(&tri.edgeC[1]);
//
//  __m256 a2 = _mm256_broadcast_ss(&tri.edgeA[2]);
//  __m256 b2 = _mm256_broadcast_ss(&tri.edgeB[2]);
//  __m256 c2 = _mm256_broadcast_ss(&tri.edgeC[2]);
//
//  __m256 edge0 = _mm256_fmadd_ps(a0, x_coords, _mm256_fmadd_ps(b0, y_coord, c0));
//  __m256 edge1 = _mm256_fmadd_ps(a1, x_coords, _mm256_fmadd_ps(b1, y_coord, c1));
//  __m256 edge2 = _mm256_fmadd_ps(a2, x_coords, _mm256_fmadd_ps(b2, y_coord, c2));
//
//  __m256 zero = _mm256_setzero_ps();
//  __m256 test0 = _mm256_cmp_ps(edge0, zero, _CMP_GE_OQ);
//  __m256 test1 = _mm256_cmp_ps(edge1, zero, _CMP_GE_OQ);
//  __m256 test2 = _mm256_cmp_ps(edge2, zero, _CMP_GE_OQ);
//
//  __m256 inside = _mm256_and_ps(test0, _mm256_and_ps(test1, test2));
//  return _mm256_castps_si256(inside);
//}

float MultiThreadedSIMDRasterizer::InterpolateDepth1x_InverseZ(float x, float y, const TransformedTriangle& tri)
{
  float alpha = tri.edgeA[0] * x + tri.edgeB[0] * y + tri.edgeC[0];
  float beta = tri.edgeA[1] * x + tri.edgeB[1] * y + tri.edgeC[1];
  float gamma = tri.edgeA[2] * x + tri.edgeB[2] * y + tri.edgeC[2];

  float inv_area = 1.0f / tri.area;
  alpha *= inv_area;
  beta *= inv_area;
  gamma *= inv_area;

  // Interpolation lineaire de 1/Z
  float interpolated_inv_z = alpha * tri.invDepths[0] + beta * tri.invDepths[1] + gamma * tri.invDepths[2];

  // Inversion pour obtenir Z final
  if (interpolated_inv_z != 0.0f)
    return 1.0f / interpolated_inv_z;

  return std::numeric_limits<float>::max(); // Protection contre division par zero
}

//void MultiThreadedSIMDRasterizer::InterpolateDepth8x(float startX, float y, const TransformedTriangle& tri, const glm::vec3& depths, const glm::vec3& wValues, float* output)
//{
//  __m256 x_coords = _mm256_set_ps(startX + 7, startX + 6, startX + 5, startX + 4, startX + 3, startX + 2, startX + 1, startX);
//  __m256 y_coord = _mm256_set1_ps(y);
//
//  __m256 inv_area = _mm256_set1_ps(1.0f / tri.area);
//
//  __m256 w0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[0]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[0]), y_coord, _mm256_broadcast_ss(&tri.edgeC[0])));
//  w0 = _mm256_mul_ps(w0, inv_area);
//
//  __m256 w1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[1]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[1]), y_coord, _mm256_broadcast_ss(&tri.edgeC[1])));
//  w1 = _mm256_mul_ps(w1, inv_area);
//
//  __m256 w2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[2]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[2]), y_coord, _mm256_broadcast_ss(&tri.edgeC[2])));
//  w2 = _mm256_mul_ps(w2, inv_area);
//
//  // Interpolation avec correction perspective
//  __m256 w_val0 = _mm256_set1_ps(wValues.x);
//  __m256 w_val1 = _mm256_set1_ps(wValues.y);
//  __m256 w_val2 = _mm256_set1_ps(wValues.z);
//
//  __m256 interp_w = _mm256_fmadd_ps(w0, w_val0, _mm256_fmadd_ps(w1, w_val1, _mm256_mul_ps(w2, w_val2)));
//
//  __m256 z_over_w0 = _mm256_div_ps(_mm256_set1_ps(depths.x), w_val0);
//  __m256 z_over_w1 = _mm256_div_ps(_mm256_set1_ps(depths.y), w_val1);
//  __m256 z_over_w2 = _mm256_div_ps(_mm256_set1_ps(depths.z), w_val2);
//
//  __m256 interp_z_over_w = _mm256_fmadd_ps(w0, z_over_w0, _mm256_fmadd_ps(w1, z_over_w1, _mm256_mul_ps(w2, z_over_w2)));
//
//  __m256 final_depth = _mm256_mul_ps(interp_z_over_w, interp_w);
//  _mm256_store_ps(output, final_depth);
//}

//void MultiThreadedSIMDRasterizer::InterpolateDepth8x_InverseZ(float startX, float y, const TransformedTriangle& tri, float* output)
//{
//  __m256 x_coords = _mm256_set_ps(startX + 7, startX + 6, startX + 5, startX + 4,    startX + 3, startX + 2, startX + 1, startX);
//  __m256 y_coord = _mm256_set1_ps(y);
//
//  __m256 inv_area = _mm256_set1_ps(1.0f / tri.area);
//
//  // ===== eTAPE 1: Calcul des coordonnees barycentriques =====
//  __m256 alpha = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[0]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[0]), y_coord, _mm256_broadcast_ss(&tri.edgeC[0])));
//  alpha = _mm256_mul_ps(alpha, inv_area);
//
//  __m256 beta = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[1]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[1]), y_coord, _mm256_broadcast_ss(&tri.edgeC[1])));
//  beta = _mm256_mul_ps(beta, inv_area);
//
//  __m256 gamma = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[2]), x_coords, _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[2]), y_coord, _mm256_broadcast_ss(&tri.edgeC[2])));
//  gamma = _mm256_mul_ps(gamma, inv_area);
//
//  // ===== eTAPE 2: Chargement des 1/Z pre-calcules =====
//  __m256 inv_z0 = _mm256_broadcast_ss(&tri.invDepths[0]);  // 1/Z0
//  __m256 inv_z1 = _mm256_broadcast_ss(&tri.invDepths[1]);  // 1/Z1
//  __m256 inv_z2 = _mm256_broadcast_ss(&tri.invDepths[2]);  // 1/Z2
//
//  // ===== eTAPE 3: Interpolation lineaire de 1/Z =====
//  __m256 interpolated_inv_z = _mm256_fmadd_ps(alpha, inv_z0, _mm256_fmadd_ps(beta, inv_z1, _mm256_mul_ps(gamma, inv_z2)));
//
//  // ===== eTAPE 4: Inversion pour obtenir Z final =====
//  __m256 one = _mm256_set1_ps(1.0f);
//  __m256 final_depth = _mm256_div_ps(one, interpolated_inv_z);
//
//  // Protection contre division par zero (cas tres rare)
//  __m256 zero = _mm256_setzero_ps();
//  __m256 is_valid = _mm256_cmp_ps(interpolated_inv_z, zero, _CMP_GT_OQ);
//  __m256 safe_depth = _mm256_set1_ps(1.0f); // Fallback depth
//  final_depth = _mm256_blendv_ps(safe_depth, final_depth, is_valid);
//
//  _mm256_store_ps(output, final_depth);
//}

void MultiThreadedSIMDRasterizer::SetupTriangleData(TransformedTriangle& tri)
{
  const auto& v0 = tri.screenVertices[0];
  const auto& v1 = tri.screenVertices[1];
  const auto& v2 = tri.screenVertices[2];

  // Edge 0: v1 -> v2
  tri.edgeA[0] = v1.y - v2.y;
  tri.edgeB[0] = v2.x - v1.x;
  tri.edgeC[0] = v1.x * v2.y - v2.x * v1.y;

  // Edge 1: v2 -> v0
  tri.edgeA[1] = v2.y - v0.y;
  tri.edgeB[1] = v0.x - v2.x;
  tri.edgeC[1] = v2.x * v0.y - v0.x * v2.y;

  // Edge 2: v0 -> v1
  tri.edgeA[2] = v0.y - v1.y;
  tri.edgeB[2] = v1.x - v0.x;
  tri.edgeC[2] = v0.x * v1.y - v1.x * v0.y;

  // Inverses des profondeurs pour interpolation
  tri.invDepths[0] = (v0.w != 0.0f) ? 1.0f / v0.w : 0.0f;
  tri.invDepths[1] = (v1.w != 0.0f) ? 1.0f / v1.w : 0.0f;
  tri.invDepths[2] = (v2.w != 0.0f) ? 1.0f / v2.w : 0.0f;
}

glm::vec4 MultiThreadedSIMDRasterizer::TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp)
{
  glm::vec4 clipSpace = mvp * glm::vec4(vertex, 1.0f);

  if (clipSpace.w != 0.0f)
  {
    clipSpace.x /= clipSpace.w;
    clipSpace.y /= clipSpace.w;
    clipSpace.z /= clipSpace.w;
  }

  float x = (clipSpace.x + 1.0f) * 0.5f * _ScreenWidth;
  float y = (1.0f - clipSpace.y) * 0.5f * _ScreenHeight;
  //float z = clipSpace.z;
  float z = (clipSpace.z + 1.0f) * 0.5f; // Map z to [0,1]

  return glm::vec4(x, y, z, clipSpace.w);
}

int MultiThreadedSIMDRasterizer::InitScene(const int nbTris)
{
  _Transformed.clear();

  Renderer::InitScene(nbTris);

  _Transformed.resize(nbTris);

  return 0;
}

void MultiThreadedSIMDRasterizer::SetTriangles(const std::vector<Triangle>& triangles)
{
  _Transformed.clear();

  Renderer::SetTriangles(triangles);

  _Transformed.resize(triangles.size());
}

void MultiThreadedSIMDRasterizer::RenderRotatingScene(float time)
{
  Clear(0xADD8E6FF);

  // Matrices de transformation
  glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0)); // Rotation Y
  glm::mat4 view = glm::lookAt(
    glm::vec3(0, 0, 3),  // Position camera
    glm::vec3(0, 0, 0),  // Point regarde
    glm::vec3(0, 1, 0)   // Up vector
  );

  glm::mat4 projection = glm::perspective(
    glm::radians(45.0f),                         // FOV
    (float)_ScreenWidth / (float)_ScreenHeight,  // Aspect ratio
    0.1f, 100.0f                                 // Near/Far planes
  );

  glm::mat4 mvp = projection * view * model;

  RenderTriangles(mvp);
}

// Fonction principale de rendu
void MultiThreadedSIMDRasterizer::RenderTriangles(const glm::mat4& mvp)
{
  // 1. Transformation des triangles
  RenderTrianglesInBatch(mvp);

  // 2. Binning des triangles aux tuiles
  BinTrianglesToTiles();

  // 3. Rendu multi-threade
  A_NextTileIndex = 0;
  A_RenderingActive = true;
  _RenderCV.notify_all();

  // Attendre que tous les threads finissent
  {
    std::unique_lock<std::mutex> lock(_TilesDoneMutex);
    _TilesDoneCV.wait(lock, [this] { return A_NextTileIndex.load() >= _Tiles.size(); });
  }

  A_RenderingActive = false;
}

// SIMD Functions Implementation - ARM NEON & AVX2 Support
// Ajouter ce code dans le fichier .cpp de MultiThreadedSIMDRasterizer

#ifdef SIMD_ARM_NEON

float getVectorElement(float32x4_t & vector, int index)
{
    switch(index) {
        case 0: return vgetq_lane_f32(vector, 0);
        case 1: return vgetq_lane_f32(vector, 1);
        case 2: return vgetq_lane_f32(vector, 2);
        case 3: return vgetq_lane_f32(vector, 3);
        default: throw std::out_of_range("Invalid vector index");
    }
}

uint32_t getVectorElement(uint32x4_t & vector, int index)
{
    switch(index) {
        case 0: return vgetq_lane_u32(vector, 0);
        case 1: return vgetq_lane_u32(vector, 1);
        case 2: return vgetq_lane_u32(vector, 2);
        case 3: return vgetq_lane_u32(vector, 3);
        default: throw std::out_of_range("Invalid vector index");
    }
}

uint32x4_t setVectorElement(uint32_t value, uint32x4_t & vector, int index)
{
    switch(index)
    {
        case 0: return vsetq_lane_u32(value, vector, 0);
        case 1: return vsetq_lane_u32(value, vector, 1);
        case 2: return vsetq_lane_u32(value, vector, 2);
        case 3: return vsetq_lane_u32(value, vector, 3);
        default: throw std::out_of_range("Invalid vector index");
    }

    return vector; // Should never reach here
}

// Test de 4 pixels en parallele avec ARM NEON
uint32x4_t MultiThreadedSIMDRasterizer::TestPixels4x_NEON(float startX, float y, const TransformedTriangle& tri)
{
  // Coordonnees des 4 pixels e tester
  float32x4_t x_coords = { startX, startX + 1.0f, startX + 2.0f, startX + 3.0f };
  float32x4_t y_coord = vdupq_n_f32(y);

  // Chargement des coefficients des 3 aretes
  float32x4_t edgeA_vec = vld1q_f32(&tri.edgeA[0]); // A0, A1, A2, 0
  float32x4_t edgeB_vec = vld1q_f32(&tri.edgeB[0]); // B0, B1, B2, 0
  float32x4_t edgeC_vec = vld1q_f32(&tri.edgeC[0]); // C0, C1, C2, 0

  // Resultats pour les 4 pixels
  uint32x4_t results = vdupq_n_u32(0);

  // Test pour chaque pixel
  for (int i = 0; i < 4; i++)
  {
    float x = getVectorElement(x_coords, i);

    // Calcul des 3 fonctions d'arete pour ce pixel
    float32x4_t x_vec = vdupq_n_f32(x);
    float32x4_t y_vec = vdupq_n_f32(y);

    // edge_values = A * x + B * y + C
    float32x4_t ax = vmulq_f32(edgeA_vec, x_vec);
    float32x4_t by = vmulq_f32(edgeB_vec, y_vec);
    float32x4_t edge_values = vaddq_f32(vaddq_f32(ax, by), edgeC_vec);

    // Test si tous les edge values sont >= 0
    uint32x4_t ge_zero = vcgeq_f32(edge_values, vdupq_n_f32(0.0f));

    // Le pixel est e l'interieur si les 3 premieres valeurs sont >= 0
    uint32_t inside = vgetq_lane_u32(ge_zero, 0) &
      vgetq_lane_u32(ge_zero, 1) &
      vgetq_lane_u32(ge_zero, 2);

    // Stocker le resultat
    results = setVectorElement(inside, results, i);
  }

  return results;
}

// Interpolation de profondeur pour 4 pixels avec ARM NEON
void MultiThreadedSIMDRasterizer::InterpolateDepth4x_InverseZ_NEON(float startX, float y, const TransformedTriangle& tri, float* output)
{
  // Coordonnees des 4 pixels
  float32x4_t x_coords = { startX, startX + 1.0f, startX + 2.0f, startX + 3.0f };
  float32x4_t y_coord = vdupq_n_f32(y);

  // Chargement des coefficients des aretes
  float32x4_t edgeA_vec = vld1q_f32(&tri.edgeA[0]);
  float32x4_t edgeB_vec = vld1q_f32(&tri.edgeB[0]);
  float32x4_t edgeC_vec = vld1q_f32(&tri.edgeC[0]);

  // Chargement des profondeurs inverses
  float32x4_t inv_depths = vld1q_f32(&tri.invDepths[0]); // 1/Z0, 1/Z1, 1/Z2, 0

  for (int i = 0; i < 4; i++) {
    float x = getVectorElement(x_coords, i);

    // Calcul des coordonnees barycentriques
    float32x4_t x_vec = vdupq_n_f32(x);
    float32x4_t y_vec = vdupq_n_f32(y);

    // edge_values = A * x + B * y + C
    float32x4_t ax = vmulq_f32(edgeA_vec, x_vec);
    float32x4_t by = vmulq_f32(edgeB_vec, y_vec);
    float32x4_t edge_values = vaddq_f32(vaddq_f32(ax, by), edgeC_vec);

    // Normalisation par l'aire du triangle
    float32x4_t barycentrics = vmulq_f32(edge_values, vdupq_n_f32(1.0f / tri.area));

    // Interpolation : 1/Z = w0 * (1/Z0) + w1 * (1/Z1) + w2 * (1/Z2)
    float32x4_t weighted_invz = vmulq_f32(barycentrics, inv_depths);

    // Somme des composants (reduction horizontale)
    float32x2_t sum_pair = vadd_f32(vget_low_f32(weighted_invz), vget_high_f32(weighted_invz));
    float interpolated_invz = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

    // Conversion en profondeur : Z = 1 / (1/Z)
    output[i] = 1.0f / interpolated_invz;
  }
}

// Rendu d'un triangle avec NEON (4 pixels par fois)
void MultiThreadedSIMDRasterizer::RenderTriangleInTile4x(const TransformedTriangle& tri, const Tile& tile)
{
  if (!tri.valid) return;

  // Bounding box du triangle dans la tuile
  float minX = std::max(tri.screenVertices[0].x, tri.screenVertices[1].x);
  minX = std::max(minX, tri.screenVertices[2].x);
  float maxX = std::min(tri.screenVertices[0].x, tri.screenVertices[1].x);
  maxX = std::min(maxX, tri.screenVertices[2].x);

  for (int i = 0; i < 3; i++) {
    minX = std::min(minX, tri.screenVertices[i].x);
    maxX = std::max(maxX, tri.screenVertices[i].x);
  }

  int startX = std::max(tile.x, static_cast<int>(std::floor(minX)));
  int endX = std::min(tile.x + tile.width - 1, static_cast<int>(std::ceil(maxX)));

  float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
  float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

  int startY = std::max(tile.y, static_cast<int>(std::floor(minY)));
  int endY = std::min(tile.y + tile.height - 1, static_cast<int>(std::ceil(maxY)));

  // Rasterisation par blocs de 4 pixels horizontalement
  for (int y = startY; y <= endY; ++y) {
    for (int x = startX; x <= endX; x += 4) {
      // Test des 4 pixels
      uint32x4_t inside_mask = TestPixels4x_NEON(static_cast<float>(x) + 0.5f,
        static_cast<float>(y) + 0.5f, tri);

      // Interpolation des profondeurs pour les 4 pixels
      float depths[4];
      InterpolateDepth4x_InverseZ_NEON(static_cast<float>(x) + 0.5f,
        static_cast<float>(y) + 0.5f, tri, depths);

      // Test de profondeur et ecriture pour chaque pixel
      for (int i = 0; i < 4 && (x + i) <= endX; ++i) {
        if (getVectorElement(inside_mask, i)) {
          int pixelX = x + i;
          int pixelIndex = y * _ScreenWidth + pixelX;

          if (depths[i] < _DepthBuffer[pixelIndex]) {
            _DepthBuffer[pixelIndex] = depths[i];
            _ColorBuffer[pixelIndex] = tri.color;
          }
        }
      }
    }
  }
}

#endif // SIMD_ARM_NEON

#ifdef SIMD_AVX2

// Version AVX2 pour comparaison (8 pixels)
__m256i MultiThreadedSIMDRasterizer::TestPixels8x_AVX2(float startX, float y, const TransformedTriangle& tri)
{
  // Coordonnees des 8 pixels
  __m256 x_coords = _mm256_set_ps(startX + 7.0f, startX + 6.0f, startX + 5.0f, startX + 4.0f,
    startX + 3.0f, startX + 2.0f, startX + 1.0f, startX);
  __m256 y_coord = _mm256_set1_ps(y);

  __m256i results = _mm256_setzero_si256();

  // Test pour chaque arete
  for (int edge = 0; edge < 3; ++edge) {
    __m256 edgeA = _mm256_set1_ps(tri.edgeA[edge]);
    __m256 edgeB = _mm256_set1_ps(tri.edgeB[edge]);
    __m256 edgeC = _mm256_set1_ps(tri.edgeC[edge]);

    // edge_value = A * x + B * y + C
    __m256 ax = _mm256_mul_ps(edgeA, x_coords);
    __m256 by = _mm256_mul_ps(edgeB, y_coord);
    __m256 edge_values = _mm256_add_ps(_mm256_add_ps(ax, by), edgeC);

    // Test >= 0
    __m256 ge_zero = _mm256_cmp_ps(edge_values, _mm256_setzero_ps(), _CMP_GE_OQ);
    __m256i ge_zero_int = _mm256_castps_si256(ge_zero);

    if (edge == 0) {
      results = ge_zero_int;
    }
    else {
      results = _mm256_and_si256(results, ge_zero_int);
    }
  }

  return results;
}

void MultiThreadedSIMDRasterizer::InterpolateDepth8x_InverseZ_AVX2(float startX, float y, const TransformedTriangle& tri, float* output)
{
  __m256 x_coords = _mm256_set_ps(startX + 7.0f, startX + 6.0f, startX + 5.0f, startX + 4.0f,
    startX + 3.0f, startX + 2.0f, startX + 1.0f, startX);
  __m256 y_coord = _mm256_set1_ps(y);
  __m256 inv_area = _mm256_set1_ps(1.0f / tri.area);

  // Calcul des coordonnees barycentriques pour les 8 pixels
  __m256 w0 = _mm256_setzero_ps(), w1 = _mm256_setzero_ps(), w2 = _mm256_setzero_ps();

  for (int edge = 0; edge < 3; ++edge) {
    __m256 edgeA = _mm256_set1_ps(tri.edgeA[edge]);
    __m256 edgeB = _mm256_set1_ps(tri.edgeB[edge]);
    __m256 edgeC = _mm256_set1_ps(tri.edgeC[edge]);

    __m256 ax = _mm256_mul_ps(edgeA, x_coords);
    __m256 by = _mm256_mul_ps(edgeB, y_coord);
    __m256 edge_values = _mm256_add_ps(_mm256_add_ps(ax, by), edgeC);
    __m256 weights = _mm256_mul_ps(edge_values, inv_area);

    if (edge == 0)
      w0 = weights;
    else if (edge == 1)
      w1 = weights;
    else
      w2 = weights;
  }

  // Interpolation des profondeurs inverses
  __m256 invZ0 = _mm256_set1_ps(tri.invDepths[0]);
  __m256 invZ1 = _mm256_set1_ps(tri.invDepths[1]);
  __m256 invZ2 = _mm256_set1_ps(tri.invDepths[2]);

  __m256 interpolated_invZ = _mm256_add_ps(
    _mm256_add_ps(_mm256_mul_ps(w0, invZ0), _mm256_mul_ps(w1, invZ1)),
    _mm256_mul_ps(w2, invZ2)
  );

  // Conversion en profondeur Z = 1 / (1/Z)
  __m256 one = _mm256_set1_ps(1.0f);
  __m256 depths = _mm256_div_ps(one, interpolated_invZ);

  _mm256_storeu_ps(output, depths);
}

#endif // SIMD_AVX2
