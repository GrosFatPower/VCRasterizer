// src/main.cpp - Compatible SFML 3.0
#include <SFML/Graphics.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <iostream>
#include <optional>
#include <memory>
#include <string>

#include "FPSCounter.h"
#include "SoftwareRasterizer.h"
#include "MultiThreadedSIMDRasterizer.h"
#include "OptimizedMultiThreadedSIMDRasterizer.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;
const int THREAD_COUNT = std::thread::hardware_concurrency();

static int S_NbTriangles = 1000;
static short S_TestNum = 1;

static bool S_Pause          = false;
static bool S_ReloadRenderer = true;
static bool S_ReloadScene    = true;
static bool S_UpdateHUD      = true;

static std::vector<std::string> S_FontPaths =
{
    "../assets/arial.ttf",                                             // Chemin relatif
    "assets/arial.ttf",                                                // Chemin depuis le build
    "/System/Library/Fonts/Arial.ttf",                                 // macOS système
    "/System/Library/Fonts/Helvetica.ttc",                             // macOS alternative
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", // Linux
    "C:/Windows/Fonts/arial.ttf"                                       // Windows
};

std::shared_ptr<Renderer> ReloadRasterizer(int testNum, int width, int height, int threadCount)
{
  if (testNum == 0)
  {
    std::cout << "Test Software Rasterizer" << std::endl;
    return std::make_shared<SoftwareRasterizer>(width, height);
  }
  else if (testNum == 1)
  {
    std::cout << "Test Multi-Threaded SIMD Rasterizer" << std::endl;
    return std::make_shared<MultiThreadedSIMDRasterizer>(width, height, threadCount);
  }
  else if (testNum == 2)
  {
    std::cout << "Test Optimized Multi-Threaded SIMD Rasterizer" << std::endl;
    return std::make_shared<OptimizedMultiThreadedSIMDRasterizer>(width, height, threadCount);
  }
  return nullptr;
}

void ManageEvents(sf::RenderWindow & window, std::shared_ptr<Renderer> rasterizer)
{
  while (std::optional<sf::Event> event = window.pollEvent())
  {
    if (event->is<sf::Event::Closed>())
      window.close();

    if (event->is<sf::Event::KeyPressed>())
    {
      auto keyEvent = event->getIf<sf::Event::KeyPressed>();
      if (keyEvent && keyEvent->code == sf::Keyboard::Key::F1)
      {
        S_TestNum = 0;
        S_ReloadRenderer = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::F2)
      {
        S_TestNum = 1;
        S_ReloadRenderer = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::F3)
      {
        S_TestNum = 2;
        S_ReloadRenderer = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::PageUp)
      {
        S_NbTriangles *= 2;
        S_ReloadScene = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::PageDown)
      {
        S_NbTriangles /= 2;
        if (S_NbTriangles < 1)
          S_NbTriangles = 1;
        S_ReloadScene = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::S && rasterizer)
      {
        rasterizer->SetEnableSIMD(!rasterizer->GetEnableSIMD());
        S_UpdateHUD = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::C && rasterizer)
      {
        rasterizer->SetBackfaceCullingEnabled(!rasterizer->GetBackfaceCullingEnabled());
        S_UpdateHUD = true;
      }
      else if (keyEvent && keyEvent->code == sf::Keyboard::Key::Space)
      {
        S_Pause = !S_Pause;
        S_UpdateHUD = true;
      }
    }
  }
}

int main()
{
  sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Vibe Coded Multi-Threaded Rasterizer");

  sf::Image image({ WIDTH, HEIGHT }, sf::Color::White);
  sf::Texture texture;
  if (!texture.loadFromImage(image))
  {
    std::cerr << "Erreur lors du chargement de la texture" << std::endl;
    return -1;
  }
  sf::Sprite sprite(texture);

  sf::Font font;
  bool fontLoaded = false;
  for (const auto& path : S_FontPaths)
  {
    if (font.openFromFile(path)) {
      std::cout << "Font loaded from: " << path << std::endl;
      fontLoaded = true;
      break;
    }
  }
  if (!fontLoaded)
    std::cout << "Warning: Unable to load any font. Using default system font." << std::endl;

  std::string HUDTextStr = "Triangle rasterizer";
  sf::Text HUD(font, HUDTextStr, 16);
  HUD.setPosition({ 5., 5. });

  std::shared_ptr<Renderer> rasterizer;

  FPSCounter fpsCounter;
  sf::Clock clock;
  float time = 0.0f;

  while ( window.isOpen() )
  {
    ManageEvents(window, rasterizer);

    if (S_ReloadRenderer)
    {
      const std::vector<Triangle> triangles = rasterizer ? rasterizer->GetTriangles() : std::vector<Triangle>();

      rasterizer = ReloadRasterizer(S_TestNum, WIDTH, HEIGHT, THREAD_COUNT);
      if (!rasterizer)
      {
        std::cout << "Unable to initialize the renderer" << std::endl;
        return -1;
      }

      if (S_TestNum == 2)
      {
        rasterizer->SetEnableSIMD(true);
        rasterizer->SetBackfaceCullingEnabled(true);
      }

      S_ReloadRenderer = false;
      if (!S_ReloadScene && (triangles.size() > 0))
        rasterizer->SetTriangles(triangles);
      else
        S_ReloadScene = true;
      S_UpdateHUD = true;
    }

    if (S_ReloadScene)
    {
      rasterizer->InitScene(S_NbTriangles);
      S_ReloadScene = false;
      S_UpdateHUD = true;
    }

    rasterizer->RenderRotatingScene(time);
    fpsCounter.update();

    const uint32_t* pixels = rasterizer->GetColorBuffer();
    texture.update(reinterpret_cast<const std::uint8_t*>(pixels));

    if (S_UpdateHUD)
    {
      if (S_TestNum == 0)
        HUDTextStr = "Single threaded Rasterizer - Press F2 for SIMD or F3 for Optimized SIMD";
      else if (S_TestNum == 1)
        HUDTextStr = "Multi-Threaded SIMD Rasterizer - Press F1 for Software or F3 for Optimized SIMD";
      else if (S_TestNum == 2)
        HUDTextStr = "Optimized Multi-Threaded SIMD Rasterizer - Press F1 for Software or Press F2 for SIMD";

      HUDTextStr += "\nNb Triangles = " + std::to_string(S_NbTriangles) + "\t(PageUp : x2, PageDown : /2)";
      HUDTextStr += "\nPause : " + std::string(S_Pause ? "ON" : "OFF") + "\t(Space : toggle ON/OFF)";
      HUDTextStr += "\nBackFace Culling : " + std::string(rasterizer->GetBackfaceCullingEnabled() ? "Enabled" : "Disabled") + "\t( C : toggle ON/OFF )";
       
      if ( (S_TestNum == 1) || (S_TestNum == 2) )
      {
        HUDTextStr += "\nSIMD : " + std::string(rasterizer->GetEnableSIMD() ? "Enabled" : "Disabled") + "\t( S : toggle ON/OFF )";
      }

      HUD.setString(HUDTextStr);

      S_UpdateHUD = false;
    }

    window.clear();
    window.draw(sprite);
    window.draw(HUD);
    window.display();

    window.setTitle("Vibe Coded Rasterizer - FPS: " + std::to_string(static_cast<int>(fpsCounter.getFPS())));

    sf::Time elapsed = clock.restart();
    float deltaTime = elapsed.asSeconds();
    if (!S_Pause)
      time += deltaTime;
  }

  return 0;
}
