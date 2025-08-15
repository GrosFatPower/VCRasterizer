#!/bin/bash

echo "Installation des dépendances sur macOS..."

# Vérifier si Homebrew est installé
if ! command -v brew &> /dev/null; then
    echo "Homebrew n'est pas installé. Installation en cours..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Installer les outils de développement si nécessaire
if ! command -v cmake &> /dev/null; then
    echo "Installation de CMake..."
    brew install cmake
fi

if ! command -v git &> /dev/null; then
    echo "Installation de Git..."
    brew install git
fi

# Installer vcpkg si nécessaire
if [ ! -d "vcpkg" ]; then
    echo "Installation de vcpkg..."
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    cd ..
fi

# Installer SFML et GLM via vcpkg
echo "Installation des bibliothèques..."
./vcpkg/vcpkg install sfml:x64-osx
./vcpkg/vcpkg install glm:x64-osx

echo "Dépendances installées !"
echo "Vous pouvez maintenant utiliser build-macos.sh pour compiler le projet."
