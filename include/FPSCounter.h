#pragma once

#include <deque>  // Changed from vector
#include <chrono>

// Version avec gestion d'evenements plus realiste
class FPSCounter
 {
public:
  FPSCounter() : lastUpdate(std::chrono::high_resolution_clock::now()) {}

  double getFPS() const { return currentFPS; }

  void update()
  {
    auto now = std::chrono::high_resolution_clock::now();
    frameTimestamps.push_back(now);

    if ( now - lastUpdate < std::chrono::duration<double>(WINDOW_SIZE) )
      return; // Pas assez de temps écoulé pour mettre à jour

    // Remove frames older than WINDOW_SIZE seconds
    auto oneSecondAgo = now - std::chrono::duration<double>(WINDOW_SIZE);
    while (!frameTimestamps.empty() && frameTimestamps.front() < oneSecondAgo)
    {
      frameTimestamps.pop_front();
    }

    // Calculate FPS based on number of frames in the last second
    currentFPS = static_cast<double>(frameTimestamps.size()) / WINDOW_SIZE;
    lastUpdate = now;
  }

  double getLastFrameTime() const
  {
    if (frameTimestamps.size() < 2)
      return 0.0;
    auto duration = std::chrono::duration<double, std::milli>(frameTimestamps.back() - *(frameTimestamps.end() - 2));
    return duration.count();
  }

private:
  std::chrono::high_resolution_clock::time_point lastUpdate;
  std::deque<std::chrono::high_resolution_clock::time_point> frameTimestamps;  // Changed to deque
  double currentFPS = 0.0;
  static constexpr double WINDOW_SIZE = 1.0; // 1 second window
};
