// src/main.cpp - Compatible SFML 3.0
#include <SFML/Graphics.hpp>
#include <iostream>
#include <optional>

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

    sf::Sprite sprite(texture);

    while (window.isOpen()) {
        // SFML 3.0: pollEvent() retourne maintenant un std::optional<sf::Event>
        while (std::optional<sf::Event> event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
