#pragma once

#include <deque>  // Changed from vector
#include <chrono>

// Version avec gestion d'événements plus réaliste
class FPSCounter {
private:
    std::chrono::high_resolution_clock::time_point lastUpdate;
    std::deque<std::chrono::high_resolution_clock::time_point> frameTimestamps;  // Changed to deque
    double currentFPS = 0.0;
    static constexpr double WINDOW_SIZE = 1.0; // 1 second window

public:
    FPSCounter() : lastUpdate(std::chrono::high_resolution_clock::now()) {}

    void update() {
        auto now = std::chrono::high_resolution_clock::now();
        frameTimestamps.push_back(now);

        // Remove frames older than WINDOW_SIZE seconds
        auto oneSecondAgo = now - std::chrono::duration<double>(WINDOW_SIZE);
        while (!frameTimestamps.empty() && frameTimestamps.front() < oneSecondAgo) {
            frameTimestamps.pop_front();
        }

        // Calculate FPS based on number of frames in the last second
        currentFPS = static_cast<double>(frameTimestamps.size()) / WINDOW_SIZE;
        lastUpdate = now;
    }

    double getFPS() const { return currentFPS; }
    
    double getLastFrameTime() const {
        if (frameTimestamps.size() < 2) return 0.0;
        auto duration = std::chrono::duration<double, std::milli>(
            frameTimestamps.back() - *(frameTimestamps.end() - 2));
        return duration.count();
    }
};
