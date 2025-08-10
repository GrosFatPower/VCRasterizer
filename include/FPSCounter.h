#pragma once

#include <vector>
#include <chrono>

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
