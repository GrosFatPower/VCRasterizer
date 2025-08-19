#include "OptimizedMultiThreadedSIMDRasterizer.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <cmath>

//-----------------------------------------------------------------------------
// CTOR
//-----------------------------------------------------------------------------
OptimizedMultiThreadedSIMDRasterizer::OptimizedMultiThreadedSIMDRasterizer(int w, int h, int numThreads)
  : Renderer(w, h)
{
  // Calcul du nombre de threads optimal
  _NumThreads = (numThreads <= 0) ? std::thread::hardware_concurrency() : numThreads;

  // Configuration des tuiles
  _TileCountX = (_ScreenWidth + TILE_SIZE - 1) / TILE_SIZE;
  _TileCountY = (_ScreenHeight + TILE_SIZE - 1) / TILE_SIZE;

  _OptimizedTiles.resize(_TileCountX * _TileCountY);

  // Initialisation des tuiles
  for (int ty = 0; ty < _TileCountY; ++ty)
  {
    for (int tx = 0; tx < _TileCountX; ++tx)
    {
      int tileIndex = ty * _TileCountX + tx;
      TileData & tileData = _OptimizedTiles[tileIndex];

      tileData._Tile.x = tx * TILE_SIZE;
      tileData._Tile.y = ty * TILE_SIZE;
      tileData._Tile.width = std::min(TILE_SIZE, _ScreenWidth - tileData._Tile.x);
      tileData._Tile.height = std::min(TILE_SIZE, _ScreenHeight - tileData._Tile.y);
      tileData._TriangleCountAtomic = 0;
      tileData._NeedsProcessingAtomic = false;

      tileData._Triangles.reserve(1000); // Reserver de l'espace
    }
  }

  // Initialisation des donnees thread-local
  _ThreadLocalData.resize(_NumThreads);
  for (int i = 0; i < _NumThreads; ++i)
    _ThreadLocalData[i] = std::make_unique<ThreadLocalData>();

  // Initialisation de la lookup table pour optimisations
  InitializeLookupTables();

  // Demarrage des threads worker
  //_ThreadWorkIndices.resize(_NumThreads);
  for (int i = 0; i < _NumThreads; ++i)
  {
    //_ThreadWorkIndices[i].store(0);
    _ThreadWorkIndices.emplace_back(0);
    _WorkerThreads.emplace_back(&OptimizedMultiThreadedSIMDRasterizer::WorkerThreadFunctionOptimized, this, i);
  }

  std::cout << "OptimizedMultiThreadedSIMDRasterizer initialized with " << _NumThreads
    << " threads, " << _TileCountX << "x" << _TileCountY << " tiles" << std::endl;
}

//-----------------------------------------------------------------------------
// DTOR
//-----------------------------------------------------------------------------
OptimizedMultiThreadedSIMDRasterizer::~OptimizedMultiThreadedSIMDRasterizer()
{
  // Arret des threads
  _ThreadsShouldRunAtomic.store(false);
  _RenderCV.notify_all();

  for (auto& thread : _WorkerThreads)
  {
    if (thread.joinable())
    {
      thread.join();
    }
  }
}

//-----------------------------------------------------------------------------
// InitializeLookupTables
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::InitializeLookupTables()
{
  // Initialisation de la lookup table pour les edge functions
  for (int i = 0; i < 256; ++i)
  {
    _EdgeLUT[i] = (float)i / 255.0f;
  }
}

//-----------------------------------------------------------------------------
// InitScene
//-----------------------------------------------------------------------------
int OptimizedMultiThreadedSIMDRasterizer::InitScene(const int nbTris)
{
  LoadTriangles(_Triangles, nbTris);
  SetTriangles(_Triangles);
  return 0;
}

//-----------------------------------------------------------------------------
// SetTriangles
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::SetTriangles(const std::vector<Triangle>& triangles)
{
  _Triangles = triangles;
  _Transformed.resize(triangles.size());
}

//-----------------------------------------------------------------------------
// RenderRotatingScene
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderRotatingScene(float time)
{
  // Matrices de transformation
  glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0));
  glm::mat4 view = glm::lookAt(
    glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)
  );
  glm::mat4 projection = glm::perspective(
    glm::radians(45.0f), (float)_ScreenWidth / (float)_ScreenHeight, 0.1f, 100.0f
  );
  glm::mat4 mvp = projection * view * model;

  // Pipeline de rendu optimise
  if ( GetEnableSIMD() )
  {
#ifdef SIMD_AVX2
    Clear8x(G_DEFAULT_COLOR); // Utilisation de la version AVX2 pour le clear
    TransformTrianglesAVX2(mvp);
#else
    Clear(G_DEFAULT_COLOR);
    TransformTriangles(mvp);
#endif
  }
  else
  {
    Clear(G_DEFAULT_COLOR);
    TransformTriangles(mvp);
  }

  HierarchicalBinning();
  RenderTrianglesMultiThreaded();
}

//-----------------------------------------------------------------------------
// Clear
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::Clear(uint32_t color)
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
          std::fill(_DepthBuffer.begin() + start, _DepthBuffer.begin() + end, G_INFINITY);
        }
      )
    );
  }

  // Attendre la fin du Clear
  for (auto& future : futures)
    future.wait();
}

//-----------------------------------------------------------------------------
// TransformVertex
//-----------------------------------------------------------------------------
glm::vec4 OptimizedMultiThreadedSIMDRasterizer::TransformVertex(const glm::vec3& vertex, const glm::mat4& mvp)
{
  glm::vec4 clipSpace = mvp * glm::vec4(vertex, 1.0f);

  // to NDC
  if (clipSpace.w != 0.0f)
  {
    clipSpace.x /= clipSpace.w;
    clipSpace.y /= clipSpace.w;
    clipSpace.z /= clipSpace.w;
  }

  // To screen space
  float x = (clipSpace.x + 1.0f) * 0.5f * _ScreenWidth;
  float y = (1.0f - clipSpace.y) * 0.5f * _ScreenHeight;
  //float z = clipSpace.z;
  float z = (clipSpace.z + 1.0f) * 0.5f; // Map z to [0,1]

  return glm::vec4(x, y, z, clipSpace.w);
}

//-----------------------------------------------------------------------------
// TransformTriangles
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::TransformTriangles(const glm::mat4& mvp)
{
  const int batchSize = 64;
  const int numBatches = ((int)_Triangles.size() + batchSize - 1) / batchSize;

  // Creation des futures pour chaque batch
  std::vector<std::future<void>> futures;
  futures.reserve(numBatches);

  for (size_t batchStart = 0; batchStart < _Triangles.size(); batchStart += batchSize)
  {
    futures.push_back(std::async(std::launch::async, [this, batchStart, batchSize, &mvp]()
      {
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
          glm::vec2 edge1 = glm::vec2(tri.screenVertices[1].x - tri.screenVertices[0].x, tri.screenVertices[1].y - tri.screenVertices[0].y);
          glm::vec2 edge2 = glm::vec2(tri.screenVertices[2].x - tri.screenVertices[0].x, tri.screenVertices[2].y - tri.screenVertices[0].y);

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
            if ((tri.screenVertices[v].x >= 0 && tri.screenVertices[v].x < _ScreenWidth)
              && (tri.screenVertices[v].y >= 0 && tri.screenVertices[v].y < _ScreenHeight))
            {
              inScreen = true;
              break;
            }
          }

          if (inScreen)
            tri.valid = true;
        }
      }
    ));
  }

  for (auto& future : futures)
    future.wait();
}

//-----------------------------------------------------------------------------
// HierarchicalBinning
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::HierarchicalBinning()
{
  // Reset des tuiles
  for (auto& tileData : _OptimizedTiles)
  {
    tileData._Triangles.clear();
    tileData._TriangleCountAtomic = 0;
    tileData._NeedsProcessingAtomic = false;
  }

  // Binning hierarchique avec frustum culling
  for (int triIndex = 0; triIndex < _Transformed.size(); ++triIndex)
  {
    const TransformedTriangle& tri = _Transformed[triIndex];
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

    // Ajout du triangle aux tuiles concernees
    for (int ty = tileMinY; ty <= tileMaxY; ++ty)
    {
      for (int tx = tileMinX; tx <= tileMaxX; ++tx)
      {
        if (tx < _TileCountX && ty < _TileCountY)
        {
          int tileIndex = ty * _TileCountX + tx;
          _OptimizedTiles[tileIndex]._Triangles.push_back(&tri);
          _OptimizedTiles[tileIndex]._TriangleCountAtomic++;
          _OptimizedTiles[tileIndex]._NeedsProcessingAtomic = true;
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
// RenderTrianglesMultiThreaded
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderTrianglesMultiThreaded()
{
  // Reset des index de travail
  _WorkStealingIndexAtomic.store(0);
  for (auto& index : _ThreadWorkIndices)
  {
    index.store(0);
  }

  // Demarrage du rendu multi-threade
  {
    std::lock_guard<std::mutex> lock(_RenderMutex);
    _RenderingActiveAtomic.store(true);
  }
  _RenderCV.notify_all();

  // Attendre que tous les threads finissent
  std::unique_lock<std::mutex> lock(_TilesDoneMutex);
  _TilesDoneCV.wait(lock, [this] {
    return _WorkStealingIndexAtomic.load() >= _OptimizedTiles.size();
    });

  _RenderingActiveAtomic.store(false);
}

//-----------------------------------------------------------------------------
// WorkerThreadFunctionOptimized
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::WorkerThreadFunctionOptimized(int threadId)
{
  ThreadLocalData* localData = _ThreadLocalData[threadId].get();

  while (_ThreadsShouldRunAtomic.load())
  {
    std::unique_lock<std::mutex> lock(_RenderMutex);
    _RenderCV.wait(lock, [this] { return _RenderingActiveAtomic.load() || !_ThreadsShouldRunAtomic.load(); });

    if (!_ThreadsShouldRunAtomic.load()) break;
    lock.unlock();

    // Work stealing
    int tileIndex;
    while ((tileIndex = _WorkStealingIndexAtomic.fetch_add(1)) < _OptimizedTiles.size())
    {
      const TileData& tileData = _OptimizedTiles[tileIndex];

      if (tileData._NeedsProcessingAtomic)
      {
        if ( GetEnableSIMD() && ( tileData._TriangleCountAtomic >= MIN_TRIANGLES_PER_TILE ) )
        {
#if defined(SIMD_AVX2)
          RenderTileAVX2(tileData._Tile, localData);
#else
          RenderTile(tileData._Tile, localData); // Fallback scalaire
#endif
        }
        else
        {
          RenderTile(tileData._Tile, localData);
        }
      }
    }

    // Notifier la fin du travail
    _TilesDoneCV.notify_one();
  }
}

//-----------------------------------------------------------------------------
// StealWork
//-----------------------------------------------------------------------------
bool OptimizedMultiThreadedSIMDRasterizer::StealWork(int threadId, int& outTileIndex)
{
  // Tentative de vol de travail depuis d'autres threads
  for (int i = 0; i < _NumThreads; ++i)
  {
    if (i == threadId)
      continue;

    int otherIndex = _ThreadWorkIndices[i].load();
    if (otherIndex < _OptimizedTiles.size())
    {
      int stolenIndex = _ThreadWorkIndices[i].fetch_add(1);
      if (stolenIndex < _OptimizedTiles.size())
      {
        outTileIndex = stolenIndex;
        return true;
      }
    }
  }
  return false;
}

//-----------------------------------------------------------------------------
// TestBlockVisibility
//-----------------------------------------------------------------------------
bool OptimizedMultiThreadedSIMDRasterizer::TestBlockVisibility(int blockX, int blockY, int blockW, int blockH, const TransformedTriangle& tri)
{
  // Test rapide de visibilite du bloc entier
  // Teste les 4 coins du bloc
  const float corners[4][2] = {
      {(float)blockX, (float)blockY},
      {(float)(blockX + blockW - 1), (float)blockY},
      {(float)blockX, (float)(blockY + blockH - 1)},
      {(float)(blockX + blockW - 1), (float)(blockY + blockH - 1)}
  };

  // Si au moins un coin est a l'interieur, le bloc est potentiellement visible
  for (int i = 0; i < 4; ++i)
  {
    if (TestPixels1x(corners[i][0] + 0.5f, corners[i][1] + 0.5f, tri))
      return true;
  }

  // Test inverse : si le triangle recouvre le bloc
  const float blockCenterX = blockX + blockW * 0.5f;
  const float blockCenterY = blockY + blockH * 0.5f;

  return TestPixels1x(blockCenterX, blockCenterY, tri);
}

//-----------------------------------------------------------------------------
// TestPixels1x
//-----------------------------------------------------------------------------
bool OptimizedMultiThreadedSIMDRasterizer::TestPixels1x(float x, float y, const TransformedTriangle& tri)
{
  const float edge0 = tri.edgeA[0] * x + tri.edgeB[0] * y + tri.edgeC[0];
  const float edge1 = tri.edgeA[1] * x + tri.edgeB[1] * y + tri.edgeC[1];
  const float edge2 = tri.edgeA[2] * x + tri.edgeB[2] * y + tri.edgeC[2];

  return (edge0 >= 0.0f) && (edge1 >= 0.0f) && (edge2 >= 0.0f);
}

//-----------------------------------------------------------------------------
// SetupTriangleData
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::SetupTriangleData(TransformedTriangle& tri)
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

  // Calcul de l'aire signee du triangle
  tri.area = tri.edgeC[0] + tri.edgeC[1] + tri.edgeC[2];

  // Verification de validite du triangle
  if (std::abs(tri.area) < G_EPSILON)
  {
    tri.valid = false;
    return;
  }

  // Back-face culling (optionnel)
  if (GetBackfaceCullingEnabled() && tri.area <= 0.0f)
  {
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

  // Pre-calcul des inverses de profondeur pour interpolation perspective-correcte
  tri.invDepths[0] = (v0.w != 0.0f) ? 1.0f / v0.w : 1.0f;
  tri.invDepths[1] = (v1.w != 0.0f) ? 1.0f / v1.w : 1.0f;
  tri.invDepths[2] = (v2.w != 0.0f) ? 1.0f / v2.w : 1.0f;
}

//-----------------------------------------------------------------------------
// InterpolateDepth1x
//-----------------------------------------------------------------------------
float OptimizedMultiThreadedSIMDRasterizer::InterpolateDepth1x(float x, float y, const TransformedTriangle& tri)
{
  // Calcul des coordonnees barycentriques
  const float u = tri.edgeA[0] * x + tri.edgeB[0] * y + tri.edgeC[0];
  const float v = tri.edgeA[1] * x + tri.edgeB[1] * y + tri.edgeC[1];
  const float w = tri.edgeA[2] * x + tri.edgeB[2] * y + tri.edgeC[2];

  // Interpolation lineaire de la profondeur Z (pour le Z-buffer)
  return u * tri.screenVertices[0].z +
    v * tri.screenVertices[1].z +
    w * tri.screenVertices[2].z;
}

//-----------------------------------------------------------------------------
// RenderTile
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderTile(const Tile& tile, ThreadLocalData* localData)
{
  const int tileIndex = (tile.y / TILE_SIZE) * _TileCountX + (tile.x / TILE_SIZE);
  const TileData& tileData = _OptimizedTiles[tileIndex];

  // Clear du tile local
  for ( int x=0; x < ( tile.width * tile.height); ++x )
  {
    localData -> _ColorBuffer[x] = G_DEFAULT_COLOR;
    localData -> _DepthBuffer[x] = G_INFINITY;
  }

  // Rendu de tous les triangles du tile
  for (const TransformedTriangle* tri : tileData._Triangles)
    RenderTriangleInTile(*tri, tile, localData);

  // Copie du tile local vers le buffer principal
  CopyTileToMainBuffer(tile, localData);
}

//-----------------------------------------------------------------------------
// CopyTileToMainBuffer
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::CopyTileToMainBuffer(const Tile& tile, ThreadLocalData* localData)
{
  for (int y = 0; y < tile.height; ++y)
  {
    const int globalY = tile.y + y;
    const int localRowStart = y * tile.width;
    const int globalRowStart = globalY * _ScreenWidth + tile.x;

    memcpy(
      &_ColorBuffer[globalRowStart],
      &localData->_ColorBuffer[localRowStart],
      tile.width * sizeof(uint32_t)
    );
  }
}

//-----------------------------------------------------------------------------
// RenderTriangleInTile
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderTriangleInTile(const TransformedTriangle& tri, const Tile& tile, ThreadLocalData* localData)
{
  // Bounding box du triangle dans la tuile
  float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
  float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
  float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

  // Intersection avec la tuile
  int startX = std::max(tile.x, (int)std::floor(minX));
  int endX   = std::min(tile.x + tile.width - 1, (int)std::ceil(maxX));
  int startY = std::max(tile.y, (int)std::floor(minY));
  int endY   = std::min(tile.y + tile.height - 1, (int)std::ceil(maxY));

  // Rasterization par blocs de pixels
  for (int y = startY; y <= endY; ++y)
  {
    for (int x = startX; x <= endX; ++x)
    {
      if ( TestPixels1x(x + 0.5f, y + 0.5f, tri) )
      {
        // Interpolation des profondeurs
        float interpolated_depth = InterpolateDepth1x(x + 0.5f, y + 0.5f, tri);

        // Z-test et ecriture des pixels
        int localX = x - tile.x;
        int localY = y - tile.y;
        int localPixelIndex = localY * tile.width + localX;
        if ( interpolated_depth < localData->_DepthBuffer[localPixelIndex] )
        {
          localData -> _DepthBuffer[localPixelIndex] = interpolated_depth;
          localData -> _ColorBuffer[localPixelIndex] = tri.color;
        }
      }
    }
  }
}

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// TransformTrianglesAVX2
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::TransformTrianglesAVX2(const glm::mat4& mvp)
{
  const size_t triangleCount = _Triangles.size();
  _Transformed.resize(triangleCount);

  // Matrice transposee pour vectorisation optimale
  alignas(32) float mvpTransposed[16];
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      mvpTransposed[i * 4 + j] = mvp[j][i];
    }
  }

  const __m256 mvpRow0 = _mm256_load_ps(&mvpTransposed[0]);
  const __m256 mvpRow1 = _mm256_load_ps(&mvpTransposed[4]);
  const __m256 mvpRow2 = _mm256_load_ps(&mvpTransposed[8]);
  const __m256 mvpRow3 = _mm256_load_ps(&mvpTransposed[12]);

  // Transformation en parallele
  for (int triIndex = 0; triIndex < triangleCount; ++triIndex)
  {
    const Triangle& tri = _Triangles[triIndex];
    TransformedTriangle& transformedTri = _Transformed[triIndex];

    // Transformer les 3 vertices du triangle
    for (int v = 0; v < 3; ++v)
    {
      glm::vec4 clipSpace = TransformVertexAVX2(tri.vertices[v], mvpRow0, mvpRow1, mvpRow2, mvpRow3);

      // Conversion vers coordonnees ecran
      if (clipSpace.w > 0.0f)
      {
        float invW = 1.0f / clipSpace.w;
        transformedTri.screenVertices[v] = glm::vec4(
          (clipSpace.x * invW + 1.0f) * 0.5f * _ScreenWidth,
          (1.0f - clipSpace.y * invW) * 0.5f * _ScreenHeight,
          clipSpace.z * invW,  // Pour Z-buffer
          clipSpace.w          // Profondeur originale pour interpolation
        );
        transformedTri.valid = true;
      }
      else
      {
        transformedTri.valid = false;
        continue;
      }
    }

    transformedTri.color = tri.color;

    if (transformedTri.valid)
    {
      SetupTriangleData(transformedTri);
    }
  }
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// Clear8x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::Clear8x(uint32_t color)
{
  // Clear vectorise
  const __m256i clearColor = _mm256_set1_epi32(color);
  const __m256 clearDepth = _mm256_set1_ps(G_INFINITY);

  const size_t pixelCount = _ScreenWidth * _ScreenHeight;
  const size_t simdPixels = (pixelCount / 8) * 8;

  // Clear en parallele
  for (int i = 0; i < simdPixels; i += 8)
  {
    _mm256_store_si256((__m256i*) & _ColorBuffer[i], clearColor);
    _mm256_store_ps(&_DepthBuffer[i], clearDepth);
  }

  // Clear des pixels restants
  for (size_t i = simdPixels; i < pixelCount; ++i)
  {
    _ColorBuffer[i] = color;
    _DepthBuffer[i] = G_INFINITY;
  }
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// RenderTileAVX2
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderTileAVX2(const Tile& tile, ThreadLocalData* localData)
{
  const int tileIndex = (tile.y / TILE_SIZE) * _TileCountX + (tile.x / TILE_SIZE);
  const TileData& tileData = _OptimizedTiles[tileIndex];

  // Clear du tile local
  const __m256i clearColor = _mm256_set1_epi32(G_DEFAULT_COLOR);
  const __m256 clearDepth = _mm256_set1_ps(G_INFINITY);

  const int tilePixels = tile.width * tile.height;
  for (int i = 0; i < tilePixels; i += 8)
  {
    if (i + 8 <= tilePixels)
    {
      _mm256_store_si256((__m256i*) & localData->_ColorBuffer[i], clearColor);
      _mm256_store_ps(&localData->_DepthBuffer[i], clearDepth);
    }
  }

  // Rendu de tous les triangles du tile
  for (const TransformedTriangle* tri : tileData._Triangles)
  {
    RenderTriangleInTile16x(*tri, tile, localData);
  }

  // Copie du tile local vers le buffer principal
  CopyTileToMainBuffer8x(tile, localData);
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// RenderTriangleInTile16x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::RenderTriangleInTile16x(const TransformedTriangle& tri, const Tile& tile, ThreadLocalData* localData)
{
  const int startX = tile.x;
  const int startY = tile.y;
  const int endX = startX + tile.width;
  const int endY = startY + tile.height;

  // Process par blocs 4x4
  for (int blockY = startY; blockY < endY; blockY += 4)
  {
    for (int blockX = startX; blockX < endX; blockX += 8)
    {

      // Test de visibilite rapide du bloc
      if (!TestBlockVisibility(blockX, blockY, 8, 4, tri))
        continue;

      // Process chaque ligne du bloc 4x4
      for (int y = blockY; y < std::min(blockY + 4, endY); ++y)
      {
        const int maxX = std::min(blockX + 8, endX);

        for (int x = blockX; x < maxX; x += 8)
        {
          // Test de 8 pixels simultanes
          __m256i mask = TestPixels8x((float)x, (float)y, tri);

          if (!_mm256_testz_si256(mask, mask))
          {
            // Calcul des profondeurs pour les 8 pixels
            alignas(32) float depths[8];
            InterpolateDepth8x((float)x, (float)y, tri, depths);

            // Mise e jour du buffer local
            UpdateLocalBuffer8x(x - startX, y - startY, tile.width, depths, mask, tri.color, localData);
          }
        }
      }
    }
  }
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// TestPixels8x
//-----------------------------------------------------------------------------
__m256i OptimizedMultiThreadedSIMDRasterizer::TestPixels8x(float startX, float y, const TransformedTriangle& tri)
{
  // Coordonnees des 8 pixels avec offset 0.5 pour le centre du pixel
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

  // Test de signes (>= 0 pour etre e l'interieur)
  const __m256 zero = _mm256_setzero_ps();
  __m256i mask0 = _mm256_castps_si256(_mm256_cmp_ps(edge0, zero, _CMP_GE_OQ));
  __m256i mask1 = _mm256_castps_si256(_mm256_cmp_ps(edge1, zero, _CMP_GE_OQ));
  __m256i mask2 = _mm256_castps_si256(_mm256_cmp_ps(edge2, zero, _CMP_GE_OQ));

  // Combinaison des masques
  return _mm256_and_si256(_mm256_and_si256(mask0, mask1), mask2);
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// InterpolateDepth8x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::InterpolateDepth8x(float startX, float y,
  const TransformedTriangle& tri, float* output)
{
  const __m256 pixelX = _mm256_set_ps(startX + 7.5f, startX + 6.5f, startX + 5.5f, startX + 4.5f,
    startX + 3.5f, startX + 2.5f, startX + 1.5f, startX + 0.5f);
  const __m256 pixelY = _mm256_set1_ps(y + 0.5f);

  // Calcul des coordonnees barycentriques
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
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// UpdateLocalBuffer8x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::UpdateLocalBuffer8x(int localX, int localY, int tileWidth,
  const float* depths, const __m256i& mask, uint32_t color, ThreadLocalData* localData)
{
  const int baseIndex = localY * tileWidth + localX;

  // Chargement des profondeurs actuelles
  __m256 currentDepths = _mm256_loadu_ps(&localData->_DepthBuffer[baseIndex]);
  __m256 newDepths = _mm256_load_ps(depths);

  // Test de profondeur
  __m256 depthMask = _mm256_cmp_ps(newDepths, currentDepths, _CMP_LT_OQ);
  __m256i finalMask = _mm256_and_si256(mask, _mm256_castps_si256(depthMask));

  // Mise e jour conditionnelle des profondeurs
  __m256 updatedDepths = _mm256_blendv_ps(currentDepths, newDepths, _mm256_castsi256_ps(finalMask));
  _mm256_storeu_ps(&localData->_DepthBuffer[baseIndex], updatedDepths);

  // Mise e jour des couleurs
  __m256i colorVec = _mm256_set1_epi32(color);
  __m256i currentColors = _mm256_loadu_si256((__m256i*) & localData->_ColorBuffer[baseIndex]);
  __m256i updatedColors = _mm256_blendv_epi8(currentColors, colorVec, finalMask);
  _mm256_storeu_si256((__m256i*) & localData->_ColorBuffer[baseIndex], updatedColors);
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// RenderTriangleInTile8x
// Versions alternatives pour comparaison de performance
//-----------------------------------------------------------------------------
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
        __m256i mask = TestPixels8x((float)x, (float)y, tri);

        if (!_mm256_testz_si256(mask, mask)) {
          alignas(32) float depths[8];
          InterpolateDepth8x((float)x, (float)y, tri, depths);
          UpdateZBuffer8x(y * _ScreenWidth + x, depths, mask, tri.color);
        }
      }
      else {
        // Traitement pixel par pixel pour les bords
        for (int px = x; px < maxX; ++px) {
          if (TestPixels1x((float)px + 0.5f, (float)y + 0.5f, tri)) {
            float depth = InterpolateDepth1x((float)px + 0.5f, (float)y + 0.5f, tri);
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
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// UpdateZBuffer8x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::UpdateZBuffer8x(int pixelIndex, const float* depths, const __m256i& mask, uint32_t color)
{
  // Chargement des profondeurs actuelles du Z-buffer
  __m256 currentDepths = _mm256_loadu_ps(&_DepthBuffer[pixelIndex]);
  __m256 newDepths = _mm256_load_ps(depths);

  // Test de profondeur vectorise
  __m256 depthMask = _mm256_cmp_ps(newDepths, currentDepths, _CMP_LT_OQ);
  __m256i finalMask = _mm256_and_si256(mask, _mm256_castps_si256(depthMask));

  // Mise e jour conditionnelle des profondeurs
  __m256 updatedDepths = _mm256_blendv_ps(currentDepths, newDepths, _mm256_castsi256_ps(finalMask));
  _mm256_storeu_ps(&_DepthBuffer[pixelIndex], updatedDepths);

  // Mise e jour conditionnelle des couleurs
  // Note: Pour 8 pixels, nous devons traiter les couleurs individuellement ou utiliser des techniques plus avancees
  alignas(32) int masks[8];
  _mm256_store_si256((__m256i*)masks, finalMask);

  for (int i = 0; i < 8; ++i) {
    if (masks[i] != 0) {
      _ColorBuffer[pixelIndex + i] = color;
    }
  }
}
#endif // SIMD_AVX2

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// CopyTileToMainBuffer8x
//-----------------------------------------------------------------------------
void OptimizedMultiThreadedSIMDRasterizer::CopyTileToMainBuffer8x(const Tile& tile, ThreadLocalData* localData)
{
  // Copie optimisee du tile local vers le buffer principal
  for (int y = 0; y < tile.height; ++y)
  {
    const int globalY = tile.y + y;
    const int localRowStart = y * tile.width;
    const int globalRowStart = globalY * _ScreenWidth + tile.x;

    // Copie vectorisee par chunks de 8 pixels
    int x = 0;
    for (; x + 8 <= tile.width; x += 8)
    {
      __m256i colors = _mm256_load_si256((__m256i*) & localData->_ColorBuffer[localRowStart + x]);
      __m256 depths = _mm256_load_ps(&localData->_DepthBuffer[localRowStart + x]);

      _mm256_storeu_si256((__m256i*) & _ColorBuffer[globalRowStart + x], colors);
      _mm256_storeu_ps(&_DepthBuffer[globalRowStart + x], depths);
    }

    // Copie des pixels restants
    for (; x < tile.width; ++x)
    {
      _ColorBuffer[globalRowStart + x] = localData->_ColorBuffer[localRowStart + x];
      _DepthBuffer[globalRowStart + x] = localData->_DepthBuffer[localRowStart + x];
    }
  }
}
#endif

#ifdef SIMD_AVX2
//-----------------------------------------------------------------------------
// TransformVertexAVX2
//-----------------------------------------------------------------------------
glm::vec4 OptimizedMultiThreadedSIMDRasterizer::TransformVertexAVX2(const glm::vec3& vertex, const __m256& mvpRow0, const __m256& mvpRow1, const __m256& mvpRow2, const __m256& mvpRow3)
{
  // Vectorisation de la multiplication matrice-vecteur
  const __m256 pos = _mm256_set_ps(0, 0, 0, 0, 1.0f, vertex.z, vertex.y, vertex.x);

  __m256 result0 = _mm256_mul_ps(pos, mvpRow0);
  __m256 result1 = _mm256_mul_ps(pos, mvpRow1);
  __m256 result2 = _mm256_mul_ps(pos, mvpRow2);
  __m256 result3 = _mm256_mul_ps(pos, mvpRow3);

  // Reduction horizontale
  result0 = _mm256_hadd_ps(result0, result1);
  result2 = _mm256_hadd_ps(result2, result3);
  result0 = _mm256_hadd_ps(result0, result2);

  alignas(32) float results[8];
  _mm256_store_ps(results, result0);

  return glm::vec4(results[0] + results[4], results[1] + results[5], results[2] + results[6], results[3] + results[7]);
}
#endif // SIMD_AVX2
