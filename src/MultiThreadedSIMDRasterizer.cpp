#include "MultiThreadedSIMDRasterizer.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

MultiThreadedSIMDRasterizer::MultiThreadedSIMDRasterizer(int w, int h, int numThreads)
  : screenWidth(w), screenHeight(h) {

  // Initialisation des compteurs de performance
  frameTimes.reserve(FPS_HISTORY_SIZE);
  lastFrameTime = std::chrono::high_resolution_clock::now();
  startTime = lastFrameTime;

  colorBuffer.resize(screenWidth * screenHeight);
  depthBuffer.resize(screenWidth * screenHeight);

  // Calcul du nombre de tuiles
  tileCountX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
  tileCountY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE;

  // Initialisation des tuiles
  tiles.reserve(tileCountX * tileCountY);
  for (int ty = 0; ty < tileCountY; ++ty) {
    for (int tx = 0; tx < tileCountX; ++tx) {
      RenderTile tile;
      tile.x = tx * TILE_SIZE;
      tile.y = ty * TILE_SIZE;
      tile.width = std::min(TILE_SIZE, screenWidth - tile.x);
      tile.height = std::min(TILE_SIZE, screenHeight - tile.y);
      tiles.push_back(tile);
    }
  }

  // Initialisation du thread pool
  int threadCount = (numThreads == 0) ? std::thread::hardware_concurrency() : numThreads;
  threadCount = std::min(threadCount, (int)tiles.size()); // Pas plus de threads que de tuiles

  for (int i = 0; i < threadCount; ++i) {
    workerThreads.emplace_back(&MultiThreadedSIMDRasterizer::workerThreadFunction, this);
  }
}

MultiThreadedSIMDRasterizer::~MultiThreadedSIMDRasterizer() {
  renderingActive = false;
  for (auto& thread : workerThreads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void MultiThreadedSIMDRasterizer::clear(uint32_t color) {
  // Clear en parallèle par chunks
  const int chunkSize = 64000; // ~256KB chunks
  int numChunks = (colorBuffer.size() + chunkSize - 1) / chunkSize;

  std::vector<std::future<void>> futures;
  for (int i = 0; i < numChunks; ++i) {
    int start = i * chunkSize;
    int end = std::min(start + chunkSize, (int)colorBuffer.size());

    futures.push_back(std::async(std::launch::async, [this, start, end, color]() {
      std::fill(colorBuffer.begin() + start, colorBuffer.begin() + end, color);
      std::fill(depthBuffer.begin() + start, depthBuffer.begin() + end,
        std::numeric_limits<float>::max());
      }));
  }

  // Attendre la fin du clear
  for (auto& future : futures) {
    future.wait();
  }
}

// Transformation et culling des triangles en batch
std::vector<TransformedTriangle> MultiThreadedSIMDRasterizer::transformTriangles(const std::vector<Triangle>& triangles,
  const glm::mat4& mvp) {
  std::vector<TransformedTriangle> transformed;
  transformed.reserve(triangles.size());

  // Traitement par batch pour optimiser le cache
  const int batchSize = 64;
  for (size_t i = 0; i < triangles.size(); i += batchSize) {
    size_t end = std::min(i + batchSize, triangles.size());

    for (size_t j = i; j < end; ++j) {
      TransformedTriangle tri;
      tri.color = triangles[j].color;
      tri.valid = false;

      // Transformation des vertices
      for (int v = 0; v < 3; ++v) {
        tri.screenVertices[v] = transformVertex(triangles[j].vertices[v], mvp);
      }

      // Backface culling
      glm::vec2 edge1 = glm::vec2(tri.screenVertices[1].x - tri.screenVertices[0].x,
        tri.screenVertices[1].y - tri.screenVertices[0].y);
      glm::vec2 edge2 = glm::vec2(tri.screenVertices[2].x - tri.screenVertices[0].x,
        tri.screenVertices[2].y - tri.screenVertices[0].y);

      float crossProduct = edge1.x * edge2.y - edge1.y * edge2.x;
      if (crossProduct <= 0) continue;

      tri.area = crossProduct * 0.5f;

      // Setup edge functions
      setupEdgeFunctions(tri);

      // Frustum culling basique
      bool inScreen = false;
      for (int v = 0; v < 3; ++v) {
        if (tri.screenVertices[v].x >= 0 && tri.screenVertices[v].x < screenWidth &&
          tri.screenVertices[v].y >= 0 && tri.screenVertices[v].y < screenHeight) {
          inScreen = true;
          break;
        }
      }

      if (inScreen) {
        tri.valid = true;
        transformed.push_back(tri);
      }
    }
  }

  return transformed;
}

// Binning des triangles par tuile (avec overlap detection)
void MultiThreadedSIMDRasterizer::binTrianglesToTiles(const std::vector<TransformedTriangle>& triangles) {
  // Clear les listes de triangles des tuiles
  for (auto& tile : tiles) {
    tile.triangles.clear();
  }

  // Pour chaque triangle, déterminer quelles tuiles il intersecte
  for (const auto& tri : triangles) {
    if (!tri.valid) continue;

    // Bounding box du triangle
    float minX = std::min({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float maxX = std::max({ tri.screenVertices[0].x, tri.screenVertices[1].x, tri.screenVertices[2].x });
    float minY = std::min({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });
    float maxY = std::max({ tri.screenVertices[0].y, tri.screenVertices[1].y, tri.screenVertices[2].y });

    // Clamp aux limites de l'écran
    minX = std::max(0.0f, minX);
    maxX = std::min((float)screenWidth - 1, maxX);
    minY = std::max(0.0f, minY);
    maxY = std::min((float)screenHeight - 1, maxY);

    // Calculer les tuiles intersectées
    int tileMinX = (int)minX / TILE_SIZE;
    int tileMaxX = (int)maxX / TILE_SIZE;
    int tileMinY = (int)minY / TILE_SIZE;
    int tileMaxY = (int)maxY / TILE_SIZE;

    // Ajouter le triangle aux tuiles concernées
    for (int ty = tileMinY; ty <= tileMaxY; ++ty) {
      for (int tx = tileMinX; tx <= tileMaxX; ++tx) {
        if (tx < tileCountX && ty < tileCountY) {
          int tileIndex = ty * tileCountX + tx;
          tiles[tileIndex].triangles.push_back(&tri);
        }
      }
    }
  }
}

// Worker thread function
void MultiThreadedSIMDRasterizer::workerThreadFunction() {
  while (true) {
    // Attendre qu'une tâche de rendu soit disponible
    if (!renderingActive.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    // Récupérer la prochaine tuile à traiter
    int tileIndex = nextTileIndex.fetch_add(1);
    if (tileIndex >= tiles.size()) {
      continue; // Plus de tuiles à traiter
    }

    renderTile(tiles[tileIndex]);
  }
}

// Rendu d'une tuile avec SIMD
void MultiThreadedSIMDRasterizer::renderTile(const RenderTile& tile) {
  for (const auto* tri : tile.triangles) {
    renderTriangleInTile(*tri, tile);
  }
}

void MultiThreadedSIMDRasterizer::renderTriangleInTile(const TransformedTriangle& tri, const RenderTile& tile) {
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

  // Alignement SIMD
  startX = (startX / 8) * 8;

  glm::vec3 depths(tri.screenVertices[0].z, tri.screenVertices[1].z, tri.screenVertices[2].z);
  glm::vec3 wValues(tri.screenVertices[0].w, tri.screenVertices[1].w, tri.screenVertices[2].w);

  // Rasterization SIMD par blocs de 8 pixels
  for (int y = startY; y <= endY; ++y) {
    for (int x = startX; x <= endX; x += 8) {
      // Test de 8 pixels simultanément
      __m256i inside_mask = testPixels8x(x + 0.5f, y + 0.5f, tri);

      if (!_mm256_testz_si256(inside_mask, inside_mask)) {
        // Interpolation des profondeurs
        SIMD_ALIGN float interpolated_depths[8];
        interpolateDepth8x(x + 0.5f, y + 0.5f, tri, depths, wValues, interpolated_depths);

        // Z-test et écriture des pixels
        uint32_t mask_array[8];
        _mm256_store_si256((__m256i*)mask_array, inside_mask);

        for (int i = 0; i < 8; ++i) {
          if (mask_array[i] != 0 && x + i < screenWidth && x + i >= 0) {
            int pixelIndex = y * screenWidth + (x + i);
            if (interpolated_depths[i] < depthBuffer[pixelIndex]) {
              depthBuffer[pixelIndex] = interpolated_depths[i];
              colorBuffer[pixelIndex] = tri.color;
              //pixelsRendered.fetch_add(1);
            }
          }
        }
      }
    }
  }

  //trianglesProcessed.fetch_add(1);
}

// Test SIMD de 8 pixels
__m256i MultiThreadedSIMDRasterizer::testPixels8x(float startX, float y, const TransformedTriangle& tri) {
  __m256 x_coords = _mm256_set_ps(startX + 7, startX + 6, startX + 5, startX + 4,
    startX + 3, startX + 2, startX + 1, startX);
  __m256 y_coord = _mm256_set1_ps(y);

  __m256 a0 = _mm256_broadcast_ss(&tri.edgeA[0]);
  __m256 b0 = _mm256_broadcast_ss(&tri.edgeB[0]);
  __m256 c0 = _mm256_broadcast_ss(&tri.edgeC[0]);

  __m256 a1 = _mm256_broadcast_ss(&tri.edgeA[1]);
  __m256 b1 = _mm256_broadcast_ss(&tri.edgeB[1]);
  __m256 c1 = _mm256_broadcast_ss(&tri.edgeC[1]);

  __m256 a2 = _mm256_broadcast_ss(&tri.edgeA[2]);
  __m256 b2 = _mm256_broadcast_ss(&tri.edgeB[2]);
  __m256 c2 = _mm256_broadcast_ss(&tri.edgeC[2]);

  __m256 edge0 = _mm256_fmadd_ps(a0, x_coords, _mm256_fmadd_ps(b0, y_coord, c0));
  __m256 edge1 = _mm256_fmadd_ps(a1, x_coords, _mm256_fmadd_ps(b1, y_coord, c1));
  __m256 edge2 = _mm256_fmadd_ps(a2, x_coords, _mm256_fmadd_ps(b2, y_coord, c2));

  __m256 zero = _mm256_setzero_ps();
  __m256 test0 = _mm256_cmp_ps(edge0, zero, _CMP_GE_OQ);
  __m256 test1 = _mm256_cmp_ps(edge1, zero, _CMP_GE_OQ);
  __m256 test2 = _mm256_cmp_ps(edge2, zero, _CMP_GE_OQ);

  __m256 inside = _mm256_and_ps(test0, _mm256_and_ps(test1, test2));
  return _mm256_castps_si256(inside);
}

void MultiThreadedSIMDRasterizer::interpolateDepth8x(float startX, float y, const TransformedTriangle& tri,
  const glm::vec3& depths, const glm::vec3& wValues, float* output) {
  __m256 x_coords = _mm256_set_ps(startX + 7, startX + 6, startX + 5, startX + 4,
    startX + 3, startX + 2, startX + 1, startX);
  __m256 y_coord = _mm256_set1_ps(y);

  __m256 inv_area = _mm256_set1_ps(1.0f / tri.area);

  __m256 w0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[0]), x_coords,
    _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[0]), y_coord,
      _mm256_broadcast_ss(&tri.edgeC[0])));
  w0 = _mm256_mul_ps(w0, inv_area);

  __m256 w1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[1]), x_coords,
    _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[1]), y_coord,
      _mm256_broadcast_ss(&tri.edgeC[1])));
  w1 = _mm256_mul_ps(w1, inv_area);

  __m256 w2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeA[2]), x_coords,
    _mm256_fmadd_ps(_mm256_broadcast_ss(&tri.edgeB[2]), y_coord,
      _mm256_broadcast_ss(&tri.edgeC[2])));
  w2 = _mm256_mul_ps(w2, inv_area);

  // Interpolation avec correction perspective
  __m256 w_val0 = _mm256_set1_ps(wValues.x);
  __m256 w_val1 = _mm256_set1_ps(wValues.y);
  __m256 w_val2 = _mm256_set1_ps(wValues.z);

  __m256 interp_w = _mm256_fmadd_ps(w0, w_val0,
    _mm256_fmadd_ps(w1, w_val1,
      _mm256_mul_ps(w2, w_val2)));

  __m256 z_over_w0 = _mm256_div_ps(_mm256_set1_ps(depths.x), w_val0);
  __m256 z_over_w1 = _mm256_div_ps(_mm256_set1_ps(depths.y), w_val1);
  __m256 z_over_w2 = _mm256_div_ps(_mm256_set1_ps(depths.z), w_val2);

  __m256 interp_z_over_w = _mm256_fmadd_ps(w0, z_over_w0,
    _mm256_fmadd_ps(w1, z_over_w1,
      _mm256_mul_ps(w2, z_over_w2)));

  __m256 final_depth = _mm256_mul_ps(interp_z_over_w, interp_w);
  _mm256_store_ps(output, final_depth);
}

void MultiThreadedSIMDRasterizer::setupEdgeFunctions(TransformedTriangle& tri) {
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
}

glm::vec4 MultiThreadedSIMDRasterizer::transformVertex(const glm::vec3& vertex, const glm::mat4& mvp) {
  glm::vec4 clipSpace = mvp * glm::vec4(vertex, 1.0f);

  if (clipSpace.w != 0.0f) {
    clipSpace.x /= clipSpace.w;
    clipSpace.y /= clipSpace.w;
    clipSpace.z /= clipSpace.w;
  }

  float x = (clipSpace.x + 1.0f) * 0.5f * screenWidth;
  float y = (1.0f - clipSpace.y) * 0.5f * screenHeight;
  float z = clipSpace.z;

  return glm::vec4(x, y, z, clipSpace.w);
}

// Fonction principale de rendu
void MultiThreadedSIMDRasterizer::renderTriangles(const std::vector<Triangle>& triangles, const glm::mat4& mvp) {
  // Reset des stats
  //trianglesProcessed = 0;
  //pixelsRendered = 0;

  auto startT = std::chrono::high_resolution_clock::now();

  // 1. Transformation des triangles
  auto transformed = transformTriangles(triangles, mvp);

  // 2. Binning des triangles aux tuiles
  binTrianglesToTiles(transformed);

  // 3. Rendu multi-threadé
  nextTileIndex = 0;
  renderingActive = true;

  // Attendre que tous les threads finissent
  while (nextTileIndex.load() < tiles.size()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  renderingActive = false;

  auto endT = std::chrono::high_resolution_clock::now();
  auto renderDuration = std::chrono::duration_cast<std::chrono::microseconds>(endT - startT);

  // Mise à jour des statistiques de frame
  updateFrameStats();

  // Stats de performance détaillées (optionnel - peut être activé pour debug)
  /*std::cout << "Frame " << getFrameCount() << " - Rendered " << trianglesProcessed.load() << " triangles, "
            << pixelsRendered.load() << " pixels in " << renderDuration.count()
            << " µs (" << workerThreads.size() << " threads)\n";*/
}

void MultiThreadedSIMDRasterizer::renderRotatingTriangle(float time)
{
  clear();

  // Définir un triangle simple
  glm::vec3 vertices[3] = {
      glm::vec3(0.0f, 1.0f, 0.0f),   // Sommet
      glm::vec3(-1.0f, -1.0f, 0.0f), // Base gauche
      glm::vec3(1.0f, -1.0f, 0.0f)   // Base droite
  };

  // Matrices de transformation
  glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0)); // Rotation Y
  glm::mat4 view = glm::lookAt(
    glm::vec3(0, 0, 5),  // Position caméra
    glm::vec3(0, 0, 0),  // Point regardé
    glm::vec3(0, 1, 0)   // Up vector
  );
  glm::mat4 projection = glm::perspective(
    glm::radians(45.0f),           // FOV
    (float)screenWidth / (float)screenHeight,  // Aspect ratio
    0.1f, 100.0f                   // Near/Far planes
  );

  glm::mat4 mvp = projection * view * model;

  std::vector<Triangle> triangles;
  triangles.emplace_back(vertices[0], vertices[1], vertices[2], 0xFF0000FF);

  renderTriangles(triangles, mvp);
}

void MultiThreadedSIMDRasterizer::renderRotatingTriangles(float time) {
  clear();

  // Créer plusieurs triangles pour tester le multi-threading
  std::vector<Triangle> triangles;

  for (int i = 0; i < 100; ++i) {
    float offset = i * 0.1f;
    glm::vec3 vertices[3] = {
        glm::vec3(sin(offset) * 0.5f, 1.0f + cos(offset) * 0.2f, offset * 0.1f),
        glm::vec3(-1.0f + sin(offset) * 0.3f, -1.0f, offset * 0.1f),
        glm::vec3(1.0f + cos(offset) * 0.3f, -1.0f, offset * 0.1f)
    };

    uint32_t color = 0xFF000000 | ((i * 25) % 255) << 16 | ((i * 50) % 255) << 8 | ((i * 75) % 255);
    triangles.emplace_back(vertices[0], vertices[1], vertices[2], color);
  }

  glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0));
  glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 8), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
  glm::mat4 projection = glm::perspective(glm::radians(45.0f),
    (float)screenWidth / (float)screenHeight, 0.1f, 100.0f);
  glm::mat4 mvp = projection * view * model;

  renderTriangles(triangles, mvp);
}

void MultiThreadedSIMDRasterizer::updateFrameStats() {
  std::lock_guard<std::mutex> lock(statsMutex);

  auto currentTime = std::chrono::high_resolution_clock::now();
  auto frameTime = std::chrono::duration<double, std::milli>(currentTime - lastFrameTime);
  lastFrameTime = currentTime;

  // Ajouter le temps de cette frame
  if (frameTimes.size() >= FPS_HISTORY_SIZE) {
    frameTimes.erase(frameTimes.begin());
  }
  frameTimes.push_back(frameTime.count());

  frameCount++;

  // Calculer FPS et temps moyen sur les dernières frames
  if (!frameTimes.empty()) {
    double totalTime = 0.0;
    for (double time : frameTimes) {
      totalTime += time;
    }
    avgFrameTime = totalTime / frameTimes.size();
    currentFPS = 1000.0 / avgFrameTime; // 1000ms / temps_en_ms = FPS
  }
}

// Getters thread-safe pour les stats
double MultiThreadedSIMDRasterizer::getCurrentFPS() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  return currentFPS;
}

double MultiThreadedSIMDRasterizer::MultiThreadedSIMDRasterizer::getAverageFrameTime() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  return avgFrameTime;
}

double MultiThreadedSIMDRasterizer::getLastFrameTime() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  return frameTimes.empty() ? 0.0 : frameTimes.back();
}

int MultiThreadedSIMDRasterizer::getFrameCount() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  return frameCount;
}

double MultiThreadedSIMDRasterizer::getTotalRenderTime() const {
  std::lock_guard<std::mutex> lock(statsMutex);
  auto currentTime = std::chrono::high_resolution_clock::now();
  auto totalTime = std::chrono::duration<double>(currentTime - startTime);
  return totalTime.count();
}

// Affichage des stats de performance détaillées
void MultiThreadedSIMDRasterizer::printPerformanceStats() const {
  std::lock_guard<std::mutex> lock(statsMutex);

  std::cout << "\n=== Performance Statistics ===\n";
  std::cout << "Current FPS: " << std::fixed << std::setprecision(1) << currentFPS << "\n";
  std::cout << "Last Frame Time: " << std::fixed << std::setprecision(2) << getLastFrameTime() << " ms\n";
  std::cout << "Average Frame Time: " << std::fixed << std::setprecision(2) << avgFrameTime << " ms\n";
  std::cout << "Total Frames: " << frameCount << "\n";
  std::cout << "Total Runtime: " << std::fixed << std::setprecision(1) << getTotalRenderTime() << " seconds\n";
  //std::cout << "Triangles Processed: " << trianglesProcessed.load() << "\n";
  //std::cout << "Pixels Rendered: " << pixelsRendered.load() << "\n";
  std::cout << "Threads: " << workerThreads.size() << "\n";
  std::cout << "==============================\n\n";
}

void MTRasterizerBenchmark::comparePerformance(int triangleCount)
{
  MultiThreadedSIMDRasterizer mtRasterizer(800, 600, 8); // 8 threads

  std::vector<Triangle> triangles;
  for (int i = 0; i < triangleCount; ++i)
  {
    float angle = i * 0.1f;
    glm::vec3 v0(sin(angle), cos(angle), 0);
    glm::vec3 v1(sin(angle + 1), cos(angle + 1), 0);
    glm::vec3 v2(sin(angle + 2), cos(angle + 2), 0);
    triangles.emplace_back(v0, v1, v2);
  }

  glm::mat4 mvp = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

  // Test multi-threadé
  auto start = std::chrono::high_resolution_clock::now();
  mtRasterizer.clear();
  mtRasterizer.renderTriangles(triangles, mvp);
  auto end = std::chrono::high_resolution_clock::now();

  auto mtDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Multi-threaded: " << mtDuration.count() << " microseconds\n";
}
