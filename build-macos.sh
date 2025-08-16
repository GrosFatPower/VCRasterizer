#!/bin/bash

echo "=== Build Script pour VCRasterizer sur macOS (Apple Silicon) ==="

# Vérifier l'architecture
ARCH=$(uname -m)
echo "Architecture détectée: $ARCH"

if [[ "$ARCH" != "arm64" ]]; then
    echo "Attention: Ce script est optimisé pour Apple Silicon (M1/M2/M3/M4)"
    echo "Votre architecture ($ARCH) utilisera les optimisations x64"
fi

# Vérifier les prérequis
echo "Vérification des prérequis..."

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake n'est pas installé. Installez-le avec: brew install cmake"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git n'est pas installé. Installez-le avec: brew install git"
    exit 1
fi

if [ ! -d "vcpkg" ]; then
    echo "❌ vcpkg n'est pas installé. Exécutez d'abord install-deps-macos.sh"
    exit 1
fi

echo "✅ Tous les prérequis sont satisfaits"

# Créer le répertoire de build
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configuration avec informations détaillées
echo ""
echo "=== Configuration du projet ==="
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de la configuration CMake"
    exit 1
fi

echo ""
echo "=== Compilation ==="
cmake --build . --config Release -j$(sysctl -n hw.ncpu)

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de la compilation"
    exit 1
fi

echo ""
echo "✅ Compilation réussie!"
echo ""

# Vérifier que l'exécutable existe
if [ -f "bin/VCRasterizer" ]; then
    echo "✅ Exécutable trouvé: bin/VCRasterizer"
    
    # Afficher des informations sur l'exécutable
    echo ""
    echo "=== Informations sur l'exécutable ==="
    file bin/VCRasterizer
    
    # Vérifier les dépendances
    echo ""
    echo "=== Dépendances dynamiques ==="
    otool -L bin/VCRasterizer
    
    echo ""
    echo "=== Test de lancement (5 secondes) ==="
    echo "Tentative de lancement de l'application..."
    timeout 5s ./bin/VCRasterizer || echo "Test de lancement terminé"
    
else
    echo "❌ Exécutable non trouvé dans bin/"
    echo "Fichiers dans le répertoire bin:"
    ls -la bin/ 2>/dev/null || echo "Répertoire bin/ inexistant"
    exit 1
fi

echo ""
echo "=== Instructions d'utilisation ==="
echo "Pour lancer l'application:"
echo "  cd build && ./bin/VCRasterizer"
echo ""
echo "Contrôles dans l'application:"
echo "  F1 : Rasterizer logiciel simple"
echo "  F2 : Rasterizer multi-thread SIMD"  
echo "  F3 : Rasterizer optimisé"
echo "  S  : Toggle SIMD on/off"
echo "  C  : Toggle backface culling"
echo "  SPACE : Pause/Resume"
echo "  PageUp/PageDown : Augmenter/Diminuer le nombre de triangles"
echo ""
echo "=== Build terminé avec succès! ==="
