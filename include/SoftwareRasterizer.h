#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class SoftwareRasterizer {

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

private:
  int width, height;
  std::vector<uint32_t> colorBuffer;  // RGBA format
  std::vector<float> depthBuffer;     // Z-buffer

  // Scene
  std::vector<Triangle> tris;

public:
  SoftwareRasterizer(int w, int h) : width(w), height(h) {
    colorBuffer.resize(width * height);
    depthBuffer.resize(width * height);
  }

  void clear(uint32_t color = 0x000000FF) {
    std::fill(colorBuffer.begin(), colorBuffer.end(), color);
    std::fill(depthBuffer.begin(), depthBuffer.end(), std::numeric_limits<float>::max());
  }

  // Transformation d'un vertex du world space vers le screen space
  glm::vec4 transformVertex(const glm::vec3& vertex, const glm::mat4& mvp) {
    glm::vec4 clipSpace = mvp * glm::vec4(vertex, 1.0f);

    // Perspective divide
    if (clipSpace.w != 0.0f) {
      clipSpace.x /= clipSpace.w;
      clipSpace.y /= clipSpace.w;
      clipSpace.z /= clipSpace.w;
    }

    // Transformation viewport (NDC [-1,1] -> screen coordinates)
    float x = (clipSpace.x + 1.0f) * 0.5f * width;
    float y = (1.0f - clipSpace.y) * 0.5f * height; // Y inversé
    float z = clipSpace.z; // Garder Z pour le depth test

    return glm::vec4(x, y, z, clipSpace.w);
  }

  // Test si un point est dans un triangle (coordonnées barycentriques)
  bool isInsideTriangle(float x, float y, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2, glm::vec3& barycentrics) {
    glm::vec2 v0v1 = v1 - v0;
    glm::vec2 v0v2 = v2 - v0;
    glm::vec2 v0p = glm::vec2(x, y) - v0;

    float dot00 = glm::dot(v0v2, v0v2);
    float dot01 = glm::dot(v0v2, v0v1);
    float dot02 = glm::dot(v0v2, v0p);
    float dot11 = glm::dot(v0v1, v0v1);
    float dot12 = glm::dot(v0v1, v0p);

    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    barycentrics.x = 1.0f - u - v; // w0
    barycentrics.y = v;            // w1
    barycentrics.z = u;            // w2

    return (u >= 0) && (v >= 0) && (u + v <= 1);
  }

  // Interpolation de la profondeur avec correction perspective
  float interpolateDepth(const glm::vec3& barycentrics, const glm::vec3& depths, const glm::vec3& wValues) {
    // Interpolation avec correction perspective
    float interpolatedW = barycentrics.x * wValues.x + barycentrics.y * wValues.y + barycentrics.z * wValues.z;
    if (interpolatedW == 0.0f) return 1.0f;

    float interpolatedZ = (barycentrics.x * depths.x / wValues.x +
      barycentrics.y * depths.y / wValues.y +
      barycentrics.z * depths.z / wValues.z) * interpolatedW;
    return interpolatedZ;
  }

  // Rasterization d'un triangle
  void drawTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::mat4& mvp, uint32_t color = 0xFF0000FF) {

    // Transformation des vertices
    glm::vec4 sv0 = transformVertex(v0, mvp);
    glm::vec4 sv1 = transformVertex(v1, mvp);
    glm::vec4 sv2 = transformVertex(v2, mvp);

    // Culling : éliminer les triangles face arrière
    glm::vec2 edge1 = glm::vec2(sv1.x - sv0.x, sv1.y - sv0.y);
    glm::vec2 edge2 = glm::vec2(sv2.x - sv0.x, sv2.y - sv0.y);
    float crossProduct = edge1.x * edge2.y - edge1.y * edge2.x;
    if (crossProduct <= 0) return; // Face arrière, on l'ignore

    // Bounding box du triangle
    int minX = std::max(0, (int)std::floor(std::min({ sv0.x, sv1.x, sv2.x })));
    int maxX = std::min(width - 1, (int)std::ceil(std::max({ sv0.x, sv1.x, sv2.x })));
    int minY = std::max(0, (int)std::floor(std::min({ sv0.y, sv1.y, sv2.y })));
    int maxY = std::min(height - 1, (int)std::ceil(std::max({ sv0.y, sv1.y, sv2.y })));

    // Rasterization
    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        glm::vec3 barycentrics;
        if (isInsideTriangle(x + 0.5f, y + 0.5f,
          glm::vec2(sv0.x, sv0.y),
          glm::vec2(sv1.x, sv1.y),
          glm::vec2(sv2.x, sv2.y),
          barycentrics)) {

          // Interpolation de la profondeur
          glm::vec3 depths(sv0.z, sv1.z, sv2.z);
          glm::vec3 wValues(sv0.w, sv1.w, sv2.w);
          float depth = interpolateDepth(barycentrics, depths, wValues);

          // Test de profondeur
          int pixelIndex = y * width + x;
          if (depth < depthBuffer[pixelIndex]) {
            depthBuffer[pixelIndex] = depth;
            colorBuffer[pixelIndex] = color;
          }
        }
      }
    }
  }

  // Rendu d'un triangle en rotation
  void renderRotatingScene(float time)
  {
    clear();

    // Matrices de transformation
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), time, glm::vec3(0, 1, 0)); // Rotation Y
    glm::mat4 view = glm::lookAt(
      glm::vec3(0, 0, 5),  // Position caméra
      glm::vec3(0, 0, 0),  // Point regardé
      glm::vec3(0, 1, 0)   // Up vector
    );
    glm::mat4 projection = glm::perspective(
      glm::radians(45.0f),           // FOV
      (float)width / (float)height,  // Aspect ratio
      0.1f, 100.0f                   // Near/Far planes
    );

    glm::mat4 mvp = projection * view * model;

    // Rendu du triangle
    for ( const auto & tri : tris ) 
      drawTriangle(tri.vertices[0], tri.vertices[1], tri.vertices[2], mvp, tri.color);
  }

  int InitSingleTriangleScene()
  {
    tris.clear();

    glm::vec3 vertices[3] = {
        glm::vec3(0.0f, 1.0f, 0.0f),   // Sommet
        glm::vec3(-1.0f, -1.0f, 0.0f), // Base gauche
        glm::vec3(1.0f, -1.0f, 0.0f)   // Base droite
    };

    tris.emplace_back(vertices[0], vertices[1], vertices[2], 0xFF0000FF);

    return 0;
  }

  int InitMultipleTrianglesScene()
  {
    tris.clear();

    // Créer plusieurs triangles pour tester le multi-threading
    for (int i = 0; i < 10000; ++i) 
    {
      float offset = i * 0.1f;
      glm::vec3 vertices[3] = {
          glm::vec3(sin(offset) * 0.5f, 1.0f + cos(offset) * 0.2f, offset * 0.1f),
          glm::vec3(-1.0f + sin(offset) * 0.3f, -1.0f, offset * 0.1f),
          glm::vec3(1.0f + cos(offset) * 0.3f, -1.0f, offset * 0.1f)
      };

      offset = static_cast <float>(rand()) / static_cast <float>(RAND_MAX) * 2.f - 1.f;
      vertices[0] += offset;
      vertices[1] += offset;
      vertices[2] += offset;

      uint32_t color = 0xFF000000 | ((i * 25) % 255) << 16 | ((i * 50) % 255) << 8 | ((i * 75) % 255);
      tris.emplace_back(vertices[0], vertices[1], vertices[2], color);
    }

    return 0;
  }

  // Accès au buffer pour affichage
  const uint32_t* getColorBuffer() const { return colorBuffer.data(); }
  int getWidth() const { return width; }
  int getHeight() const { return height; }
};
