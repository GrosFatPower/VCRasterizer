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

std::unique_ptr<Renderer> ReloadRasterizer(int testNum, int width, int height, int threadCount)
{
  if (testNum == 0)
  {
    std::cout << "Test Software Rasterizer" << std::endl;
    return std::make_unique<SoftwareRasterizer>(width, height);
  }
  else if (testNum == 1)
  {
    std::cout << "Test Multi-Threaded SIMD Rasterizer" << std::endl;
    return std::make_unique<MultiThreadedSIMDRasterizer>(width, height, threadCount);
  }
  else if (testNum == 2)
  {
    std::cout << "Test Optimized Multi-Threaded SIMD Rasterizer" << std::endl;
    return std::make_unique<OptimizedMultiThreadedSIMDRasterizer>(width, height, threadCount);
  }
  return nullptr;
}

int main()
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

  sf::Font font;
  if (!font.openFromFile("../assets/arial.ttf"))
  {
    std::cout << "Unable to load arial.ttf" << std::endl;
    return -1;
  }

  std::string headerTextStr = "Triangle rasterizer";
  sf::Text headerText(font, headerTextStr, 16);
  headerText.setPosition({5., 5.});

  std::string nbTrisTextStr = "Nb Triangles = " + std::to_string(S_NbTriangles) + " (PageUp : x2, PageDown : /2)";
  sf::Text nbTrisText(font, nbTrisTextStr, 16);
  nbTrisText.setPosition({ 5., 25. });

  std::string optionsTextStr = "S : toggle SIMD, C : toggle backFace culling, SPACE : toggle pause";
  sf::Text optionsText(font, optionsTextStr, 16);
  optionsText.setPosition({ 5., 45. });

  sf::Sprite sprite(texture);

  std::unique_ptr<Renderer> rasterizer;

  FPSCounter fpsCounter;
  sf::Clock clock;
  float time = 0.0f;
  bool pause = false;

  bool reloadRenderer = true;
  bool reloadScene = true;

  while (window.isOpen())
  {
    // SFML 3.0: pollEvent() retourne maintenant un std::optional<sf::Event>
    while (std::optional<sf::Event> event = window.pollEvent())
    {
      if (event->is<sf::Event::Closed>())
        window.close();

      if (event->is<sf::Event::KeyPressed>())
      {
        auto keyEvent = event -> getIf<sf::Event::KeyPressed>();
        if (keyEvent && keyEvent->code == sf::Keyboard::Key::F1)
        {
          S_TestNum = 0;
          reloadRenderer = true;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::F2)
        {
          S_TestNum = 1;
          reloadRenderer = true;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::F3)
        {
          S_TestNum = 2;
          reloadRenderer = true;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::PageUp)
        {
          S_NbTriangles *= 2;
          reloadScene = true;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::PageDown)
        {
          S_NbTriangles /= 2;
          if (S_NbTriangles < 1)
            S_NbTriangles = 1;
          reloadScene = true;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::S && rasterizer)
        {
          rasterizer -> SetEnableSIMD(!rasterizer->GetEnableSIMD());
          if ( rasterizer -> GetEnableSIMD() )
            std::cout << "SIMD : Enabled" << std::endl;
          else
            std::cout << "SIMD : Disabled" << std::endl;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::C && rasterizer)
        {
          rasterizer->SetBackfaceCullingEnabled(!rasterizer->GetBackfaceCullingEnabled());
          if (rasterizer -> GetBackfaceCullingEnabled())
            std::cout << "BackfaceCulling : Enabled" << std::endl;
          else
            std::cout << "BackfaceCulling : Disabled" << std::endl;
        }
        else if (keyEvent && keyEvent->code == sf::Keyboard::Key::Space)
        {
          pause = !pause;
          if (pause)
            std::cout << "Paused" << std::endl;
          else
            std::cout << "Resumed" << std::endl;
        }
      }
    }

    if ( reloadRenderer )
    {
      const std::vector<Triangle> triangles = rasterizer ? rasterizer->GetTriangles() : std::vector<Triangle>();

      rasterizer = ReloadRasterizer(S_TestNum, WIDTH, HEIGHT, THREAD_COUNT);
      if (!rasterizer)
      {
        std::cout << "Unable to initialize the renderer" << std::endl;
        return -1;
      }

      if ( S_TestNum == 0 )
        headerTextStr = "Single threaded Rasterizer - Press F2 for SIMD or F3 for Optimized SIMD";
      else if (S_TestNum == 1 )
        headerTextStr = "Multi-Threaded SIMD Rasterizer - Press F1 for Software or F3 for Optimized SIMD";
      else if (S_TestNum == 2)
      {
        headerTextStr = "Optimized Multi-Threaded SIMD Rasterizer - Press F1 for Software or Press F2 for SIMD";
        //rasterizer -> SetRenderMode(RenderMode::AVX2);
        rasterizer -> SetEnableSIMD(true);
        rasterizer -> SetBackfaceCullingEnabled(true);
      }
      headerText = sf::Text(font, headerTextStr, 16);
      headerText.setPosition({ 5., 5. });

      reloadRenderer = false;
      if ( !reloadScene && ( triangles.size() > 0 ) )
        rasterizer -> SetTriangles(triangles);
      else
        reloadScene = true;
    }

    if ( reloadScene )
    {
      nbTrisTextStr = "Nb Triangles = " + std::to_string(S_NbTriangles) + " (PageUp : x2, PageDown : /2)";
      nbTrisText = sf::Text(font, nbTrisTextStr, 16);
      nbTrisText.setPosition({ 5., 25. });

      rasterizer -> InitScene(S_NbTriangles);
      reloadScene = false;
    }

    rasterizer -> RenderRotatingScene(time);
    fpsCounter.update();

    const uint32_t* pixels = rasterizer -> GetColorBuffer();
    texture.update(reinterpret_cast<const std::uint8_t*>(pixels));

    window.clear();
    window.draw(sprite);
    window.draw(headerText);
    window.draw(nbTrisText);
    if ( S_TestNum == 1 )
      window.draw(optionsText);
    window.display();

    window.setTitle("Vibe Coded Rasterizer - FPS: " + std::to_string(static_cast<int>(fpsCounter.getFPS())));

    sf::Time elapsed = clock.restart();
    float deltaTime = elapsed.asSeconds();
    if ( !pause)
      time += deltaTime;
  }

  return 0;
}
