// OptimizedMultiThreadedSIMDRasterizer.cpp - Implémentation complète
#include "OptimizedMultiThreadedSIMDRasterizer.h"
#include <algorithm>
#include <iostream>
#include <cmath>

OptimizedMultiThreadedSIMDRasterizer::OptimizedMultiThreadedSIMDRasterizer(int w, int h, int numThreads)
  : Renderer(w, h)
{
  // Calcul du nombre de threads optimal
  _NumThreads = (numThreads <= 0) ? std::thread::hardware_concurrency() : numThreads;
  _NumThreads = std::min(_NumThreads, 16); // Limiter à 16 threads max

  // Configuration des tuiles
  _TileCountX = (_ScreenWidth + TILE_SIZE - 1) / TILE_SIZE;
  _TileCountY = (_ScreenHeight + TILE_SIZE - 1) / TILE_SIZE;

  _OptimizedTiles.resize(_TileCountX * _TileCountY);

  // Initialisation des tuiles
  for (int ty = 0; ty < _TileCountY; ++ty) {
    for (int tx = 0; tx < _TileCountX; ++tx) {
      int tileIndex = ty * _TileCountX + tx;
      TileData& tileData = _OptimizedTiles[tileIndex];

      tileData.tile.x = tx * TILE_SIZE;
      tileData.tile.y = ty * TILE_SIZE;
      tileData.tile.width = std::min(TILE_SIZE, _ScreenWidth - tileData.tile.x);
      tileData.tile.height = std::min(TILE_SIZE, _ScreenHeight - tileData.tile.y);
      tileData.triangleCount = 0;
      tileData.needsProcessing = false;

      tileData.triangles.reserve(1000); // Réserver de l'espace
    }
  }

  // Initialisation des données thread-local
  _ThreadLocalData.resize(_NumThreads);
  for (int i = 0; i < _NumThreads; ++i) {
    _ThreadLocalData[i] = std::make_unique<ThreadLocalData>();
    _ThreadLocalData[i]->localTriangles.reserve(1000);
  }

  // Initialisation de la lookup table pour optimisations
  InitializeLookupTables();

  // Démarrage des threads worker
  //_ThreadWorkIndices.resize(_NumThreads);
  for (int i = 0; i < _NumThreads; ++i) {
    //_ThreadWorkIndices[i].store(0);
    _ThreadWorkIndices.emplace_back(0);
    _WorkerThreads.emplace_back(&OptimizedMultiThreadedSIMDRasterizer::WorkerThreadFunctionOptimized, this, i);
  }

  std::cout << "OptimizedMultiThreadedSIMDRasterizer initialized with " << _NumThreads
    << " threads, " << _TileCountX << "x" << _TileCountY << " tiles" << std::endl;
}

OptimizedMultiThreadedSIMDRasterizer::~OptimizedMultiThreadedSIMDRasterizer()
{
  // Arrêt des threads
  _ThreadsShouldRun.store(false);
  _RenderCV.notify_all();

  for (auto& thread : _WorkerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void OptimizedMultiThreadedSIMDRasterizer::InitializeLookupTables()
{
  // Initialisation de la lookup table pour les edge functions
  for (int i = 0; i < 256; ++i) {
    _EdgeLUT[i] = (float)i / 255.0f;
  }
}

int OptimizedMultiThreadedSIMDRasterizer::InitScene(const int nbTris)
{
  LoadTriangles(_Triangles, nbTris);
  SetTriangles(_Triangles);
  return 0;
}

void OptimizedMultiThreadedSIMDRasterizer::SetTriangles(const std::vector<Triangle>& triangles)
{
  _Triangles = triangles;
  _Transformed.resize(triangles.size());
}

void OptimizedMultiThreadedSIMDRasterizer::RenderRotatingScene(float time)
{
  Clear(0x87CEEBFF);

  // Matrices de transformation
  glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0));
  glm::mat4 view = glm::lookAt(
    glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)
  );
  glm::mat4 projection = glm::perspective(
    glm::radians(45.0f), (float)_ScreenWidth / (float)_ScreenHeight, 0.1f, 100.0f
  );
  glm::mat4 mvp = projection * view * model;

  // Pipeline de rendu optimisé
  TransformTrianglesVectorized(mvp);
  HierarchicalBinning();
  RenderTrianglesMultiThreaded();
}

void OptimizedMultiThreadedSIMDRasterizer::Clear(uint32_t color)
{
  // Clear vectorisé
  const __m256i clearColor = _mm256_set1_epi32(color);
  const __m256 clearDepth = _mm256_set1_ps(std::numeric_limits<float>::max());

  const size_t pixelCount = _ScreenWidth * _ScreenHeight;
  const size_t simdPixels = (pixelCount / 8) * 8;

  // Clear en parallèle
#pragma omp parallel for
  for (int i = 0; i < simdPixels; i += 8) {
    _mm256_store_si256((__m256i*) & _ColorBuffer[i], clearColor);
    _mm256_store_ps(&_DepthBuffer[i], clearDepth);
  }

  // Clear des pixels restants
  for (size_t i = simdPixels; i < pixelCount; ++i) {
    _ColorBuffer[i] = color;
    _DepthBuffer[i] = std::numeric_limits<float>::max();
  }
}

void OptimizedMultiThreadedSIMDRasterizer::TransformTrianglesVectorized(const glm::mat4& mvp)
{
  const size_t triangleCount = _Triangles.size();
  _Transformed.resize(triangleCount);

  // Matrice transposée pour vectorisation optimale
  alignas(32) float mvpTransposed[16];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      mvpTransposed[i * 4 + j] = mvp[j][i];
    }
  }

  const __m256 mvpRow0 = _mm256_load_ps(&mvpTransposed[0]);
  const __m256 mvpRow1 = _mm256_load_ps(&mvpTransposed[4]);
  const __m256 mvpRow2 = _mm256_load_ps(&mvpTransposed[8]);
  const __m256 mvpRow3 = _mm256_load_ps(&mvpTransposed[12]);

  // Transformation en parallèle
#pragma omp parallel for
  for (int triIndex = 0; triIndex < triangleCount; ++triIndex) {
    const Triangle& tri = _Triangles[triIndex];
    TransformedTriangle& transformedTri = _Transformed[triIndex];

    // Transformer les 3 vertices du triangle
    for (int v = 0; v < 3; ++v) {
      glm::vec4 clipSpace = TransformVertexSIMD(tri.vertices[v], mvpRow0, mvpRow1, mvpRow2, mvpRow3);

      // Conversion vers coordonnées écran
      if (clipSpace.w > 0.0f) {
        float invW = 1.0f / clipSpace.w;
        transformedTri.screenVertices[v] = glm::vec4(
          (clipSpace.x * invW + 1.0f) * 0.5f * _ScreenWidth,
          (1.0f - clipSpace.y * invW) * 0.5f * _ScreenHeight,
          clipSpace.z * invW,  // Pour Z-buffer
          clipSpace.w          // Profondeur originale pour interpolation
        );
      }
      else {
        transformedTri.valid = false;
        continue;
      }
    }

    transformedTri.color = tri.color;

    if (transformedTri.valid) {
      SetupTriangleDataOptimized(transformedTri);
    }
  }
}

glm::vec4 OptimizedMultiThreadedSIMDRasterizer::TransformVertexSIMD(const glm::vec3& vertex,
  const __m256& mvpRow0, const __m256& mvpRow1, const __m256& mvpRow2, const __m256& mvpRow3)
{
  // Vectorisation de la multiplication matrice-vecteur
  const __m256 pos = _mm256_set_ps(0, 0, 0, 0, 1.0f, vertex.z, vertex.y, vertex.x);

  __m256 result0 = _mm256_mul_ps(pos, mvpRow0);
  __m256 result1 = _mm256_mul_ps(pos, mvpRow1);
  __m256 result2 = _mm256_mul_ps(pos, mvpRow2);
  __m256 result3 = _mm256_mul_ps(pos, mvpRow3);

  // Réduction horizontale
  result0 = _mm256_hadd_ps(result0, result1);
  result2 = _mm256_hadd_ps(result2, result3);
  result0 = _mm256_hadd_ps(result0, result2);

  alignas(32) float results[8];
  _mm256_store_ps(results, result0);

  return glm::vec4(results[0] + results[4], results[1] + results[5],
    results[2] + results[6], results[3] + results[7]);
}

void OptimizedMultiThreadedSIMDRasterizer::HierarchicalBinning()
{
  // Reset des tuiles
  for (auto& tileData : _OptimizedTiles) {
    tileData.triangles.clear();
    tileData.triangleCount = 0;
    tileData.needsProcessing = false;
  }

  // Binning hiérarchique avec frustum culling
#pragma omp parallel for
  for (int triIndex = 0; triIndex < _Transformed.size(); ++triIndex) {
    const TransformedTriangle& tri = _Transformed[triIndex];
    if (!tri.valid) continue;

    // Calcul de la bounding box du triangle
    float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
    float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

    // Frustum culling
    if (maxX < 0 || minX >= _ScreenWidth || maxY < 0 || minY >= _ScreenHeight) {
      continue;
    }

    // Calcul des tuiles affectées
    int tileMinX = std::max(0, (int)(minX / TILE_SIZE));
    int tileMaxX = std::min(_TileCountX - 1, (int)(maxX / TILE_SIZE));
    int tileMinY = std::max(0, (int)(minY / TILE_SIZE));
    int tileMaxY = std::min(_TileCountY - 1, (int)(maxY / TILE_SIZE));

    // Ajout du triangle aux tuiles concernées
    for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
      for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
        int tileIndex = ty * _TileCountX + tx;

#pragma omp critical
        {
          _OptimizedTiles[tileIndex].triangles.push_back(&tri);
          _OptimizedTiles[tileIndex].triangleCount++;
          _OptimizedTiles[tileIndex].needsProcessing = true;
        }
      }
    }
  }
}

void OptimizedMultiThreadedSIMDRasterizer::RenderTrianglesMultiThreaded()
{
  // Reset des index de travail
  _WorkStealingIndex.store(0);
  for (auto& index : _ThreadWorkIndices) {
    index.store(0);
  }

  // Démarrage du rendu multi-threadé
  {
    std::lock_guard<std::mutex> lock(_RenderMutex);
    _RenderingActive.store(true);
  }
  _RenderCV.notify_all();

  // Attendre que tous les threads finissent
  std::unique_lock<std::mutex> lock(_TilesDoneMutex);
  _TilesDoneCV.wait(lock, [this] {
    return _WorkStealingIndex.load() >= _OptimizedTiles.size();
    });

  _RenderingActive.store(false);
}

void OptimizedMultiThreadedSIMDRasterizer::WorkerThreadFunctionOptimized(int threadId)
{
  ThreadLocalData* localData = _ThreadLocalData[threadId].get();

  while (_ThreadsShouldRun.load()) {
    std::unique_lock<std::mutex> lock(_RenderMutex);
    _RenderCV.wait(lock, [this] { return _RenderingActive.load() || !_ThreadsShouldRun.load(); });

    if (!_ThreadsShouldRun.load()) break;
    lock.unlock();

    // Work stealing
    int tileIndex;
    while ((tileIndex = _WorkStealingIndex.fetch_add(1)) < _OptimizedTiles.size()) {
      const TileData& tileData = _OptimizedTiles[tileIndex];

      if (tileData.needsProcessing && tileData.triangleCount >= MIN_TRIANGLES_PER_TILE) {
        RenderTileAVX2(tileData.tile, localData);
      }
    }

    // Notifier la fin du travail
    _TilesDoneCV.notify_one();
  }
}

bool OptimizedMultiThreadedSIMDRasterizer::StealWork(int threadId, int& outTileIndex)
{
  // Tentative de vol de travail depuis d'autres threads
  for (int i = 0; i < _NumThreads; ++i) {
    if (i == threadId) continue;

    int otherIndex = _ThreadWorkIndices[i].load();
    if (otherIndex < _OptimizedTiles.size()) {
      int stolenIndex = _ThreadWorkIndices[i].fetch_add(1);
      if (stolenIndex < _OptimizedTiles.size()) {
        outTileIndex = stolenIndex;
        return true;
      }
    }
  }
  return false;
}

void OptimizedMultiThreadedSIMDRasterizer::RenderTileAVX2(const Tile& tile, ThreadLocalData* localData)
{
  const int tileIndex = (tile.y / TILE_SIZE) * _TileCountX + (tile.x / TILE_SIZE);
  const TileData& tileData = _OptimizedTiles[tileIndex];

  // Clear du tile local
  const __m256i clearColor = _mm256_set1_epi32(0x87CEEBFF);
  const __m256 clearDepth = _mm256_set1_ps(std::numeric_limits<float>::max());

  const int tilePixels = tile.width * tile.height;
  for (int i = 0; i < tilePixels; i += 8) {
    if (i + 8 <= tilePixels) {
      _mm256_store_si256((__m256i*) & localData->colorBuffer[i], clearColor);
      _mm256_store_ps(&localData->depthBuffer[i], clearDepth);
    }
  }

  // Rendu de tous les triangles du tile
  for (const TransformedTriangle* tri : tileData.triangles) {
    RenderTriangleInTile16x(*tri, tile, localData);
  }

  // Copie du tile local vers le buffer principal
  CopyTileToMainBuffer(tile, localData);
}

void OptimizedMultiThreadedSIMDRasterizer::RenderTriangleInTile16x(const TransformedTriangle& tri,
  const Tile& tile, ThreadLocalData* localData)
{
  const int startX = tile.x;
  const int startY = tile.y;
  const int endX = startX + tile.width;
  const int endY = startY + tile.height;

  // Process par blocs 4x4
  for (int blockY = startY; blockY < endY; blockY += 4) {
    for (int blockX = startX; blockX < endX; blockX += 8) {

      // Test de visibilité rapide du bloc
      if (!TestBlockVisibility(blockX, blockY, 8, 4, tri)) {
        continue;
      }

      // Process chaque ligne du bloc 4x4
      for (int y = blockY; y < std::min(blockY + 4, endY); ++y) {
        const int maxX = std::min(blockX + 8, endX);

        for (int x = blockX; x < maxX; x += 8) {
          // Test de 8 pixels simultanés
          __m256i mask = TestPixels8xOptimized((float)x, (float)y, tri);

          if (!_mm256_testz_si256(mask, mask)) {
            // Calcul des profondeurs pour les 8 pixels
            alignas(32) float depths[8];
            InterpolateDepth8x_InverseZ((float)x, (float)y, tri, depths);

            // Mise à jour du buffer local
            UpdateLocalBuffer8x(x - startX, y - startY, tile.width,
              depths, mask, tri.color, localData);
          }
        }
      }
    }
  }
}

__m256i OptimizedMultiThreadedSIMDRasterizer::TestPixels8xOptimized(float startX, float y, const TransformedTriangle& tri)
{
  // Coordonnées des 8 pixels avec offset 0.5 pour le centre du pixel
  const __m256 pixelX = _mm256_set_ps(startX + 7.5f, startX + 6.5f, startX + 5.5f, startX + 4.5f,
    startX + 3.5f, startX + 2.5f, startX + 1.5f, startX + 0.5f);
  const __m256 pixelY = _mm256_set1_ps(y + 0.5f);

  // Chargement des coefficients d'edge function
  const __m256 edgeA0 = _mm256_broadcast_ss(&tri.edgeA[0]);
  const __m256 edgeB0 = _mm256_broadcast_ss(&tri.edgeB[0]);
  const __m256 edgeC0 = _mm256_broadcast_ss(&tri.edgeC[0]);

  const __m256 edgeA1 = _mm256_broadcast_ss(&tri.edgeA[1]);
  const __m256 edgeB1 = _mm256_broadcast_ss(&tri.edgeB[1]);
  const __m256 edgeC1 = _mm256_broadcast_ss(&tri.edgeC[1]);

  const __m256 edgeA2 = _mm256_broadcast_ss(&tri.edgeA[2]);
  const __m256 edgeB2 = _mm256_broadcast_ss(&tri.edgeB[2]);
  const __m256 edgeC2 = _mm256_broadcast_ss(&tri.edgeC[2]);

  // Calcul des 3 edge functions avec FMA
  __m256 edge0 = _mm256_fmadd_ps(pixelX, edgeA0, _mm256_fmadd_ps(pixelY, edgeB0, edgeC0));
  __m256 edge1 = _mm256_fmadd_ps(pixelX, edgeA1, _mm256_fmadd_ps(pixelY, edgeB1, edgeC1));
  __m256 edge2 = _mm256_fmadd_ps(pixelX, edgeA2, _mm256_fmadd_ps(pixelY, edgeB2, edgeC2));

  // Test de signes (>= 0 pour être à l'intérieur)
  const __m256 zero = _mm256_setzero_ps();
  __m256i mask0 = _mm256_castps_si256(_mm256_cmp_ps(edge0, zero, _CMP_GE_OQ));
  __m256i mask1 = _mm256_castps_si256(_mm256_cmp_ps(edge1, zero, _CMP_GE_OQ));
  __m256i mask2 = _mm256_castps_si256(_mm256_cmp_ps(edge2, zero, _CMP_GE_OQ));

  // Combinaison des masques
  return _mm256_and_si256(_mm256_and_si256(mask0, mask1), mask2);
}

void OptimizedMultiThreadedSIMDRasterizer::InterpolateDepth8x_InverseZ(float startX, float y,
  const TransformedTriangle& tri, float* output)
{
  const __m256 pixelX = _mm256_set_ps(startX + 7.5f, startX + 6.5f, startX + 5.5f, startX + 4.5f,
    startX + 3.5f, startX + 2.5f, startX + 1.5f, startX + 0.5f);
  const __m256 pixelY = _mm256_set1_ps(y + 0.5f);

  // Calcul des coordonnées barycentriques
  const __m256 edgeA0 = _mm256_broadcast_ss(&tri.edgeA[0]);
  const __m256 edgeB0 = _mm256_broadcast_ss(&tri.edgeB[0]);
  const __m256 edgeC0 = _mm256_broadcast_ss(&tri.edgeC[0]);

  const __m256 edgeA1 = _mm256_broadcast_ss(&tri.edgeA[1]);
  const __m256 edgeB1 = _mm256_broadcast_ss(&tri.edgeB[1]);
  const __m256 edgeC1 = _mm256_broadcast_ss(&tri.edgeC[1]);

  const __m256 edgeA2 = _mm256_broadcast_ss(&tri.edgeA[2]);
  const __m256 edgeB2 = _mm256_broadcast_ss(&tri.edgeB[2]);
  const __m256 edgeC2 = _mm256_broadcast_ss(&tri.edgeC[2]);

  __m256 u = _mm256_fmadd_ps(pixelX, edgeA0, _mm256_fmadd_ps(pixelY, edgeB0, edgeC0));
  __m256 v = _mm256_fmadd_ps(pixelX, edgeA1, _mm256_fmadd_ps(pixelY, edgeB1, edgeC1));
  __m256 w = _mm256_fmadd_ps(pixelX, edgeA2, _mm256_fmadd_ps(pixelY, edgeB2, edgeC2));

  // Interpolation des profondeurs Z
  const __m256 z0 = _mm256_broadcast_ss(&tri.screenVertices[0].z);
  const __m256 z1 = _mm256_broadcast_ss(&tri.screenVertices[1].z);
  const __m256 z2 = _mm256_broadcast_ss(&tri.screenVertices[2].z);

  __m256 interpolatedZ = _mm256_fmadd_ps(u, z0, _mm256_fmadd_ps(v, z1, _mm256_mul_ps(w, z2)));

  _mm256_store_ps(output, interpolatedZ);
}

void OptimizedMultiThreadedSIMDRasterizer::UpdateLocalBuffer8x(int localX, int localY, int tileWidth,
  const float* depths, const __m256i& mask, uint32_t color, ThreadLocalData* localData)
{
  const int baseIndex = localY * tileWidth + localX;

  // Chargement des profondeurs actuelles
  __m256 currentDepths = _mm256_loadu_ps(&localData->depthBuffer[baseIndex]);
  __m256 newDepths = _mm256_load_ps(depths);

  // Test de profondeur
  __m256 depthMask = _mm256_cmp_ps(newDepths, currentDepths, _CMP_LT_OQ);
  __m256i finalMask = _mm256_and_si256(mask, _mm256_castps_si256(depthMask));

  // Mise à jour conditionnelle des profondeurs
  __m256 updatedDepths = _mm256_blendv_ps(currentDepths, newDepths, _mm256_castsi256_ps(finalMask));
  _mm256_storeu_ps(&localData->depthBuffer[baseIndex], updatedDepths);

  // Mise à jour des couleurs
  __m256i colorVec = _mm256_set1_epi32(color);
  __m256i currentColors = _mm256_loadu_si256((__m256i*) & localData->colorBuffer[baseIndex]);
  __m256i updatedColors = _mm256_blendv_epi8(currentColors, colorVec, finalMask);
  _mm256_storeu_si256((__m256i*) & localData->colorBuffer[baseIndex], updatedColors);
}

void OptimizedMultiThreadedSIMDRasterizer::CopyTileToMainBuffer(const Tile& tile, ThreadLocalData* localData)
{
  // Copie optimisée du tile local vers le buffer principal
  for (int y = 0; y < tile.height; ++y) {
    const int globalY = tile.y + y;
    const int localRowStart = y * tile.width;
    const int globalRowStart = globalY * _ScreenWidth + tile.x;

    // Copie vectorisée par chunks de 8 pixels
    int x = 0;
    for (; x + 8 <= tile.width; x += 8) {
      __m256i colors = _mm256_load_si256((__m256i*) & localData->colorBuffer[localRowStart + x]);
      __m256 depths = _mm256_load_ps(&localData->depthBuffer[localRowStart + x]);

      _mm256_storeu_si256((__m256i*) & _ColorBuffer[globalRowStart + x], colors);
      _mm256_storeu_ps(&_DepthBuffer[globalRowStart + x], depths);
    }

    // Copie des pixels restants
    for (; x < tile.width; ++x) {
      _ColorBuffer[globalRowStart + x] = localData->colorBuffer[localRowStart + x];
      _DepthBuffer[globalRowStart + x] = localData->depthBuffer[localRowStart + x];
    }
  }
}

bool OptimizedMultiThreadedSIMDRasterizer::TestBlockVisibility(int blockX, int blockY, int blockW, int blockH,
  const TransformedTriangle& tri)
{
  // Test rapide de visibilité du bloc entier
  // Teste les 4 coins du bloc
  const float corners[4][2] = {
      {(float)blockX, (float)blockY},
      {(float)(blockX + blockW - 1), (float)blockY},
      {(float)blockX, (float)(blockY + blockH - 1)},
      {(float)(blockX + blockW - 1), (float)(blockY + blockH - 1)}
  };

  // Si au moins un coin est à l'intérieur, le bloc est potentiellement visible
  for (int i = 0; i < 4; ++i) {
    if (TestPixels1x(corners[i][0] + 0.5f, corners[i][1] + 0.5f, tri)) {
      return true;
    }
  }

  // Test inverse : si le triangle recouvre le bloc
  const float blockCenterX = blockX + blockW * 0.5f;
  const float blockCenterY = blockY + blockH * 0.5f;

  return TestPixels1x(blockCenterX, blockCenterY, tri);
}

bool OptimizedMultiThreadedSIMDRasterizer::TestPixels1x(float x, float y, const TransformedTriangle& tri)
{
  const float edge0 = tri.edgeA[0] * x + tri.edgeB[0] * y + tri.edgeC[0];
  const float edge1 = tri.edgeA[1] * x + tri.edgeB[1] * y + tri.edgeC[1];
  const float edge2 = tri.edgeA[2] * x + tri.edgeB[2] * y + tri.edgeC[2];

  return (edge0 >= 0.0f) && (edge1 >= 0.0f) && (edge2 >= 0.0f);
}

void OptimizedMultiThreadedSIMDRasterizer::SetupTriangleDataOptimized(TransformedTriangle& tri)
{
  const glm::vec4& v0 = tri.screenVertices[0];
  const glm::vec4& v1 = tri.screenVertices[1];
  const glm::vec4& v2 = tri.screenVertices[2];

  // Calcul des coefficients des edge functions
  tri.edgeA[0] = v1.y - v2.y;
  tri.edgeA[1] = v2.y - v0.y;
  tri.edgeA[2] = v0.y - v1.y;

  tri.edgeB[0] = v2.x - v1.x;
  tri.edgeB[1] = v0.x - v2.x;
  tri.edgeB[2] = v1.x - v0.x;

  tri.edgeC[0] = v1.x * v2.y - v2.x * v1.y;
  tri.edgeC[1] = v2.x * v0.y - v0.x * v2.y;
  tri.edgeC[2] = v0.x * v1.y - v1.x * v0.y;

  // Calcul de l'aire signée du triangle
  tri.area = tri.edgeC[0] + tri.edgeC[1] + tri.edgeC[2];

  // Vérification de validité du triangle
  if (std::abs(tri.area) < 1e-6f) {
    tri.valid = false;
    return;
  }

  // Back-face culling (optionnel)
  if (GetBackfaceCullingEnabled() && tri.area <= 0.0f) {
    tri.valid = false;
    return;
  }

  tri.valid = true;

  // Normalisation des edge functions par l'aire
  const float invArea = 1.0f / tri.area;
  for (int i = 0; i < 3; ++i) {
    tri.edgeA[i] *= invArea;
    tri.edgeB[i] *= invArea;
    tri.edgeC[i] *= invArea;
  }

  // Pré-calcul des inverses de profondeur pour interpolation perspective-correcte
  tri.invDepths[0] = (v0.w != 0.0f) ? 1.0f / v0.w : 1.0f;
  tri.invDepths[1] = (v1.w != 0.0f) ? 1.0f / v1.w : 1.0f;
  tri.invDepths[2] = (v2.w != 0.0f) ? 1.0f / v2.w : 1.0f;
}

// Méthodes additionnelles pour compléter l'interface Renderer
void OptimizedMultiThreadedSIMDRasterizer::SetEnableSIMD(bool enable)
{
  _EnableSIMD = enable;
  std::cout << "SIMD " << (enable ? "Enabled" : "Disabled") << std::endl;
}

bool OptimizedMultiThreadedSIMDRasterizer::GetEnableSIMD() const
{
  return _EnableSIMD;
}

void OptimizedMultiThreadedSIMDRasterizer::SetBackfaceCullingEnabled(bool enable)
{
  _BackfaceCullingEnabled = enable;
  std::cout << "Backface Culling " << (enable ? "Enabled" : "Disabled") << std::endl;
}

bool OptimizedMultiThreadedSIMDRasterizer::GetBackfaceCullingEnabled() const
{
  return _BackfaceCullingEnabled;
}

// Versions alternatives pour comparaison de performance
void OptimizedMultiThreadedSIMDRasterizer::RenderTriangleInTile8x(const TransformedTriangle& tri, const Tile& tile)
{
  const int startX = tile.x;
  const int startY = tile.y;
  const int endX = startX + tile.width;
  const int endY = startY + tile.height;

  // Version 8x plus simple pour comparaison
  for (int y = startY; y < endY; ++y) {
    for (int x = startX; x < endX; x += 8) {
      const int maxX = std::min(x + 8, endX);
      const int pixelCount = maxX - x;

      if (pixelCount == 8) {
        // Traitement SIMD complet de 8 pixels
        __m256i mask = TestPixels8xOptimized((float)x, (float)y, tri);

        if (!_mm256_testz_si256(mask, mask)) {
          alignas(32) float depths[8];
          InterpolateDepth8x_InverseZ((float)x, (float)y, tri, depths);
          UpdateZBuffer8x(y * _ScreenWidth + x, depths, mask, tri.color);
        }
      }
      else {
        // Traitement pixel par pixel pour les bords
        for (int px = x; px < maxX; ++px) {
          if (TestPixels1x((float)px + 0.5f, (float)y + 0.5f, tri)) {
            float depth = InterpolateDepth1x_InverseZ((float)px + 0.5f, (float)y + 0.5f, tri);
            int pixelIndex = y * _ScreenWidth + px;

            if (depth < _DepthBuffer[pixelIndex]) {
              _DepthBuffer[pixelIndex] = depth;
              _ColorBuffer[pixelIndex] = tri.color;
            }
          }
        }
      }
    }
  }
}

float OptimizedMultiThreadedSIMDRasterizer::InterpolateDepth1x_InverseZ(float x, float y, const TransformedTriangle& tri)
{
  // Calcul des coordonnées barycentriques
  const float u = tri.edgeA[0] * x + tri.edgeB[0] * y + tri.edgeC[0];
  const float v = tri.edgeA[1] * x + tri.edgeB[1] * y + tri.edgeC[1];
  const float w = tri.edgeA[2] * x + tri.edgeB[2] * y + tri.edgeC[2];

  // Interpolation linéaire de la profondeur Z (pour le Z-buffer)
  return u * tri.screenVertices[0].z +
    v * tri.screenVertices[1].z +
    w * tri.screenVertices[2].z;
}

void OptimizedMultiThreadedSIMDRasterizer::UpdateZBuffer8x(int pixelIndex, const float* depths, const __m256i& mask, uint32_t color)
{
  // Chargement des profondeurs actuelles du Z-buffer
  __m256 currentDepths = _mm256_loadu_ps(&_DepthBuffer[pixelIndex]);
  __m256 newDepths = _mm256_load_ps(depths);

  // Test de profondeur vectorisé
  __m256 depthMask = _mm256_cmp_ps(newDepths, currentDepths, _CMP_LT_OQ);
  __m256i finalMask = _mm256_and_si256(mask, _mm256_castps_si256(depthMask));

  // Mise à jour conditionnelle des profondeurs
  __m256 updatedDepths = _mm256_blendv_ps(currentDepths, newDepths, _mm256_castsi256_ps(finalMask));
  _mm256_storeu_ps(&_DepthBuffer[pixelIndex], updatedDepths);

  // Mise à jour conditionnelle des couleurs
  // Note: Pour 8 pixels, nous devons traiter les couleurs individuellement ou utiliser des techniques plus avancées
  alignas(32) int masks[8];
  _mm256_store_si256((__m256i*)masks, finalMask);

  for (int i = 0; i < 8; ++i) {
    if (masks[i] != 0) {
      _ColorBuffer[pixelIndex + i] = color;
    }
  }
}

// Méthodes de debugging et profiling
void OptimizedMultiThreadedSIMDRasterizer::PrintPerformanceStats() const
{
  std::cout << "\n=== PERFORMANCE STATISTICS ===" << std::endl;
  std::cout << "Triangles processed: " << _TrianglesProcessed.load() << std::endl;
  std::cout << "Pixels rasterized: " << _PixelsRasterized.load() << std::endl;
  std::cout << "Number of tiles: " << _OptimizedTiles.size() << std::endl;
  std::cout << "Active threads: " << _NumThreads << std::endl;

  // Calcul du pourcentage de tuiles actives
  int activeTiles = 0;
  for (const auto& tile : _OptimizedTiles) {
    if (tile.needsProcessing) activeTiles++;
  }

  float tileUtilization = (float)activeTiles / _OptimizedTiles.size() * 100.0f;
  std::cout << "Tile utilization: " << tileUtilization << "%" << std::endl;
}

void OptimizedMultiThreadedSIMDRasterizer::ResetPerformanceCounters()
{
  _TrianglesProcessed.store(0);
  _PixelsRasterized.store(0);
}

// Gestion des différents modes de rendu
void OptimizedMultiThreadedSIMDRasterizer::SetRenderMode(RenderMode mode)
{
  _CurrentRenderMode = mode;

  switch (mode) {
  case RenderMode::SCALAR:
    std::cout << "Render mode: Scalar (no SIMD)" << std::endl;
    break;
  case RenderMode::SSE:
    std::cout << "Render mode: SSE (4-wide SIMD)" << std::endl;
    break;
  case RenderMode::AVX2:
    std::cout << "Render mode: AVX2 (8-wide SIMD)" << std::endl;
    break;
  case RenderMode::AVX512:
    std::cout << "Render mode: AVX512 (16-wide SIMD)" << std::endl;
    break;
  }
}

// Méthode de benchmark interne
double OptimizedMultiThreadedSIMDRasterizer::BenchmarkRenderTime(int frames)
{
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < frames; ++i) {
    RenderRotatingScene(i * 0.016f);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  return duration.count() / 1000.0; // Retour en millisecondes
}

// Validation de la cohérence des résultats
bool OptimizedMultiThreadedSIMDRasterizer::ValidateResults(const OptimizedMultiThreadedSIMDRasterizer& reference)
{
  const uint32_t* refColorBuffer = reference.GetColorBuffer();
  const float* refDepthBuffer = reference.GetDepthBuffer();

  int pixelDifferences = 0;
  int depthDifferences = 0;
  const int totalPixels = _ScreenWidth * _ScreenHeight;

  for (int i = 0; i < totalPixels; ++i) {
    if (_ColorBuffer[i] != refColorBuffer[i]) {
      pixelDifferences++;
    }

    if (std::abs(_DepthBuffer[i] - refDepthBuffer[i]) > 1e-5f) {
      depthDifferences++;
    }
  }

  const float colorAccuracy = 1.0f - (float)pixelDifferences / totalPixels;
  const float depthAccuracy = 1.0f - (float)depthDifferences / totalPixels;

  std::cout << "Validation Results:" << std::endl;
  std::cout << "Color accuracy: " << (colorAccuracy * 100.0f) << "%" << std::endl;
  std::cout << "Depth accuracy: " << (depthAccuracy * 100.0f) << "%" << std::endl;

  return (colorAccuracy > 0.99f) && (depthAccuracy > 0.99f);
}
