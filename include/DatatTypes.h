#pragma once

#include <vector>
#include <random>
#include <glm/glm.hpp>

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

  // Pre-computed inverse depths for linear interpolation (1/Z method)
  SIMD_ALIGN float invDepths[3];  // 1/Z0, 1/Z1, 1/Z2
};

struct Tile {
  int x, y;
  int width, height;
  std::vector<const TransformedTriangle*> triangles;
};

inline int LoadTriangles(std::vector<Triangle> & oTriangles, const int nbTris)
{
  oTriangles.clear();

  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> posDist(-1.0f, 1.0f);      // For centroid x/y
  std::uniform_real_distribution<float> zDist(-1.0f, 1.0f);         // For centroid z (depth)
  std::uniform_real_distribution<float> sizeDist(0.05f, 0.25f);    // Triangle size
  std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.14159265f); // Orientation

  for (int i = 0; i < nbTris; ++i)
  {
    // Random centroid in NDC
    float cx = posDist(gen);
    float cy = posDist(gen);
    float cz = zDist(gen);

    // Random size and orientation
    float size = sizeDist(gen);
    float baseAngle = angleDist(gen);

    glm::vec3 vertices[3];
    for (int v = 0; v < 3; ++v)
    {
      float angle = baseAngle + v * (2.0f * 3.14159265f / 3.0f);
      float r = size * (0.8f + 0.4f * posDist(gen)); // Slightly irregular triangle
      vertices[v] = glm::vec3(
        cx + r * std::cos(angle),
        cy + r * std::sin(angle),
        cz + size * (0.1f + 0.5f * posDist(gen)) // Random depth offset
      );
    }

    uint32_t color = 0xFF000000 | ((i * 25) % 255) << 16 | ((i * 50) % 255) << 8 | ((i * 75) % 255);
    oTriangles.emplace_back(vertices[0], vertices[1], vertices[2], color);
  }

  return 0;
}
