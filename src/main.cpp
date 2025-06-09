// src/main.cpp - Compatible SFML 3.0
#include <SFML/Graphics.hpp>
#include <iostream>
#include <optional>

#include "SoftwareRasterizer.h"

int main() {
    const int WIDTH = 800;
    const int HEIGHT = 600;

    // SFML 3.0: VideoMode prend maintenant un sf::Vector2u
    sf::RenderWindow window(sf::VideoMode({ WIDTH, HEIGHT }), "Vibe Coded Rasterizer");

    // SFML 3.0: Utiliser le constructeur d'Image au lieu de create()
    sf::Image image({ WIDTH, HEIGHT }, sf::Color::Red);

    sf::Texture texture;
    if (!texture.loadFromImage(image)) {
        std::cerr << "Erreur lors du chargement de la texture" << std::endl;
        return -1;
    }

    SoftwareRasterizer rasterizer(800, 600);

    float time = 0.0f;
    const float deltaTime = 0.016f; // ~60 FPS

    sf::Sprite sprite(texture);

    while (window.isOpen()) {
        // SFML 3.0: pollEvent() retourne maintenant un std::optional<sf::Event>
        while (std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        rasterizer.renderRotatingTriangle(time);

        const uint32_t* pixels = rasterizer.getColorBuffer();
        texture.update(reinterpret_cast<const std::uint8_t*>(pixels));

        window.clear();
        window.draw(sprite);
        window.display();

        time += deltaTime;
    }

    return 0;
}
