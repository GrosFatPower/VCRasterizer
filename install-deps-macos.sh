#!/bin/bash

echo "Installation des dépendances sur macOS..."

# Vérifier si Homebrew est installé
if ! command -v brew &> /dev/null; then
    echo "Homebrew n'est pas installé. Installation en cours..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Installer les outils de développement nécessaires
echo "Installation des outils de développement..."

if ! command -v cmake &> /dev/null; then
    echo "Installation de CMake..."
    brew install cmake
fi

if ! command -v git &> /dev/null; then
    echo "Installation de Git..."
    brew install git
fi

if ! command -v pkg-config &> /dev/null; then
    echo "Installation de pkg-config..."
    brew install pkg-config
fi

# Détection de l'architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    VCPKG_TARGET="arm64-osx"
    echo "Architecture détectée: ARM64 (Apple Silicon)"
else
    VCPKG_TARGET="x64-osx"
    echo "Architecture détectée: x64 (Intel)"
fi

# Installer vcpkg si nécessaire
if [ ! -d "vcpkg" ]; then
    echo "Installation de vcpkg..."
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    cd ..
else
    echo "vcpkg déjà installé, mise à jour..."
    cd vcpkg
    git pull
    cd ..
fi

# Configuration du triplet par défaut
export VCPKG_DEFAULT_TRIPLET=$VCPKG_TARGET

# Installer SFML et GLM via vcpkg avec le bon triplet
echo "Installation des bibliothèques pour $VCPKG_TARGET..."
echo "Cela peut prendre quelques minutes..."

# Nettoyer d'éventuelles installations précédentes avec le mauvais triplet
echo "Nettoyage des installations précédentes..."
./vcpkg/vcpkg remove --outdated --recurse

# Installation avec le bon triplet
echo "Installation de SFML..."
./vcpkg/vcpkg install sfml:$VCPKG_TARGET

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation de SFML"
    echo "Essayons avec une approche alternative..."
    
    # Alternative via Homebrew si vcpkg échoue
    echo "Installation de SFML via Homebrew en fallback..."
    brew install sfml
    
    # Créer un lien symbolique pour que CMake trouve SFML
    if [ ! -d "/usr/local/lib/cmake/SFML" ] && [ -d "/opt/homebrew/lib/cmake/SFML" ]; then
        sudo mkdir -p /usr/local/lib/cmake
        sudo ln -sf /opt/homebrew/lib/cmake/SFML /usr/local/lib/cmake/SFML
    fi
fi

echo "Installation de GLM..."
./vcpkg/vcpkg install glm:$VCPKG_TARGET

# Vérification des installations
echo ""
echo "=== Vérification des installations ==="
echo "Packages installés:"
./vcpkg/vcpkg list

echo ""
echo "✅ Dépendances installées pour $VCPKG_TARGET!"
echo "Vous pouvez maintenant utiliser build-macos.sh pour compiler le projet."
