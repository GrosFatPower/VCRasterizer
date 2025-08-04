// src/main.cpp - Compatible SFML 3.0
#include <SFML/Graphics.hpp>
#include <iostream>
#include <optional>

#include "SoftwareRasterizer.h"
#include "MultiThreadedSIMDRasterizer.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;

// Version avec gestion d'événements plus réaliste
class FPSCounter {
private:
  std::chrono::high_resolution_clock::time_point lastUpdate;
  std::vector<double> frameTimes;
  double currentFPS = 0.0;
  static constexpr int HISTORY_SIZE = 30;

public:
  FPSCounter() : lastUpdate(std::chrono::high_resolution_clock::now()) {
    frameTimes.reserve(HISTORY_SIZE);
  }

  void update() {
    auto now = std::chrono::high_resolution_clock::now();
    auto deltaTime = std::chrono::duration<double, std::milli>(now - lastUpdate);
    lastUpdate = now;

    if (frameTimes.size() >= HISTORY_SIZE) {
      frameTimes.erase(frameTimes.begin());
    }
    frameTimes.push_back(deltaTime.count());

    if (!frameTimes.empty()) {
      double avgTime = 0.0;
      for (double time : frameTimes) {
        avgTime += time;
      }
      avgTime /= frameTimes.size();
      currentFPS = 1000.0 / avgTime;
    }
  }

  double getFPS() const { return currentFPS; }
  double getLastFrameTime() const {
    return frameTimes.empty() ? 0.0 : frameTimes.back();
  }
};


int TestSoftwareRasterizer()
{
  // SFML 3.0: VideoMode prend maintenant un sf::Vector2u
  sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Vibe Coded Rasterizer");
  window.setVerticalSyncEnabled(false);

  // SFML 3.0: Utiliser le constructeur d'Image au lieu de create()
  sf::Image image({ WIDTH, HEIGHT }, sf::Color::Red);

  sf::Texture texture;
  if (!texture.loadFromImage(image)) {
    std::cerr << "Erreur lors du chargement de la texture" << std::endl;
    return -1;
  }

  SoftwareRasterizer rasterizer(WIDTH, HEIGHT);

  //rasterizer.InitSingleTriangleScene();
  rasterizer.InitMultipleTrianglesScene();

  FPSCounter fpsCounter;
  sf::Clock clock;
  float time = 0.0f;
  //const float deltaTime = 0.016f; // ~60 FPS
  //bool showDetailedStats = false;

  sf::Sprite sprite(texture);

  while ( window.isOpen() )
  {
    // SFML 3.0: pollEvent() retourne maintenant un std::optional<sf::Event>
    while (std::optional<sf::Event> event = window.pollEvent())
    {
      if (event->is<sf::Event::Closed>())
        window.close();
    }

    rasterizer.renderRotatingScene(time);
    fpsCounter.update();

    const uint32_t* pixels = rasterizer.getColorBuffer();
    texture.update(reinterpret_cast<const std::uint8_t*>(pixels));

    window.clear();
    window.draw(sprite);
    window.display();

    // Affichage FPS compact en continu
    //std::cout << "FPS: " << std::setw(6) << std::fixed << std::setprecision(1)
    //  << fpsCounter.getFPS()
    //  << " | " << std::setw(6) << std::fixed << std::setprecision(2)
    //  << fpsCounter.getLastFrameTime() << "ms | "
    //  //<< "Triangles: " << rasterizer.trianglesProcessed.load()
    //  << "     \r" << std::flush;

    window.setTitle("Vibe Coded Rasterizer - FPS: " + std::to_string(static_cast<int>(fpsCounter.getFPS())));

    // Stats détaillées si activées
    //if (showDetailedStats && rasterizer.getFrameCount() % 60 == 0) {
    //  std::cout << std::endl;
    //  rasterizer.printPerformanceStats();
    //}

    sf::Time elapsed = clock.restart();
    float deltaTime = elapsed.asSeconds();
    time += deltaTime;
  }

  return 0;
}

int TestMultiThreadedSIMDRasterizer()
{
  // SFML 3.0: VideoMode prend maintenant un sf::Vector2u
  sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Vibe Coded Multi-Threaded Rasterizer");

  // SFML 3.0: Utiliser le constructeur d'Image au lieu de create()
  sf::Image image({ WIDTH, HEIGHT }, sf::Color::White);
  sf::Texture texture;
  if (!texture.loadFromImage(image))
  {
    std::cerr << "Erreur lors du chargement de la texture" << std::endl;
    return -1;
  }

  int threadCount = std::thread::hardware_concurrency();
  MultiThreadedSIMDRasterizer rasterizer(WIDTH, HEIGHT, threadCount);

  //rasterizer.InitSingleTriangleScene();
  const int nbTris = 1000;
  rasterizer.InitMultipleTrianglesScene(nbTris);

  FPSCounter fpsCounter;
  sf::Clock clock;
  float time = 0.0f;
  //const float deltaTime = 0.016f; // ~60 FPS
  //bool showDetailedStats = false;

  sf::Sprite sprite(texture);
  while ( window.isOpen() )
  {
    // SFML 3.0: pollEvent() retourne maintenant un std::optional<sf::Event>
    while (std::optional<sf::Event> event = window.pollEvent())
    {
      if (event->is<sf::Event::Closed>())
        window.close();
    }
    rasterizer.RenderRotatingScene(time);
    fpsCounter.update();

    const uint32_t* pixels = rasterizer.GetColorBuffer();
    texture.update(reinterpret_cast<const std::uint8_t*>(pixels));

    window.clear();
    window.draw(sprite);
    window.display();

    // Affichage FPS compact en continu
    //std::cout << "FPS: " << std::setw(6) << std::fixed << std::setprecision(1)
    //  << fpsCounter.getFPS()
    //  << " | " << std::setw(6) << std::fixed << std::setprecision(2)
    //  << fpsCounter.getLastFrameTime() << "ms | "
    //  //<< "Triangles: " << rasterizer.trianglesProcessed.load()
    //  << "     \r" << std::flush;

    window.setTitle("Vibe Coded Rasterizer - FPS: " + std::to_string(static_cast<int>(fpsCounter.getFPS())));

    // Stats détaillées si activées
    //if (showDetailedStats && rasterizer.getFrameCount() % 60 == 0) {
    //  std::cout << std::endl;
    //  rasterizer.printPerformanceStats();
    //}

    sf::Time elapsed = clock.restart();
    float deltaTime = elapsed.asSeconds();
    time += deltaTime;
  }
  return 0;
}

int main()
{
  static short testNum = 1;

  if ( testNum == 0 )
  {
    std::cout << "Test Software Rasterizer" << std::endl;
    testNum++;
    return TestSoftwareRasterizer();
  }
  else if ( testNum == 1 )
  {
    std::cout << "Test Multi-Threaded SIMD Rasterizer" << std::endl;
    testNum++;
    return TestMultiThreadedSIMDRasterizer();
  }

  return 1;
}
