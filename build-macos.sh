#!/bin/bash

# Créer le répertoire de build s'il n'existe pas
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configurer le projet avec CMake
echo "Configuration du projet..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compiler le projet
echo "Compilation du projet..."
cmake --build . --config Release

echo "Compilation terminée !"
echo "L'exécutable se trouve dans build/bin/"
