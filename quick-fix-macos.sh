#!/bin/bash

echo "=== Script de correction rapide pour les dépendances ==="

# Installer pkg-config immédiatement
if ! command -v pkg-config &> /dev/null; then
    echo "Installation de pkg-config..."
    brew install pkg-config
fi

# Nettoyer l'installation vcpkg précédente avec le mauvais triplet
if [ -d "vcpkg" ]; then
    echo "Nettoyage des installations x64-osx..."
    ./vcpkg/vcpkg remove sfml:x64-osx --recurse || true
    ./vcpkg/vcpkg remove --outdated --recurse || true
fi

# Détection de l'architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    VCPKG_TARGET="arm64-osx"
    echo "✅ Architecture ARM64 détectée - utilisation d'arm64-osx"
else
    VCPKG_TARGET="x64-osx"
    echo "✅ Architecture x64 détectée - utilisation d'x64-osx"
fi

# Réinstaller avec le bon triplet
echo "Installation de SFML pour $VCPKG_TARGET..."
./vcpkg/vcpkg install sfml:$VCPKG_TARGET

if [ $? -eq 0 ]; then
    echo "✅ SFML installé avec succès!"
else
    echo "❌ Échec de l'installation de SFML via vcpkg"
    echo "Installation alternative via Homebrew..."
    brew install sfml
    
    # Créer un fichier de configuration pour CMake
    mkdir -p cmake
    cat > cmake/FindSFML.cmake << 'EOF'
# Fallback SFML finder for Homebrew installation
if(APPLE)
    set(SFML_ROOT "/opt/homebrew" "/usr/local")
    find_path(SFML_INCLUDE_DIR SFML/Config.hpp
        HINTS ${SFML_ROOT}
        PATH_SUFFIXES include)
    
    find_library(SFML_SYSTEM_LIBRARY
        NAMES sfml-system sfml-system-s
        HINTS ${SFML_ROOT}
        PATH_SUFFIXES lib)
        
    find_library(SFML_WINDOW_LIBRARY
        NAMES sfml-window sfml-window-s
        HINTS ${SFML_ROOT}
        PATH_SUFFIXES lib)
        
    find_library(SFML_GRAPHICS_LIBRARY
        NAMES sfml-graphics sfml-graphics-s
        HINTS ${SFML_ROOT}
        PATH_SUFFIXES lib)
    
    if(SFML_INCLUDE_DIR AND SFML_SYSTEM_LIBRARY AND SFML_WINDOW_LIBRARY AND SFML_GRAPHICS_LIBRARY)
        set(SFML_FOUND TRUE)
        add_library(SFML::System UNKNOWN IMPORTED)
        set_target_properties(SFML::System PROPERTIES IMPORTED_LOCATION ${SFML_SYSTEM_LIBRARY})
        target_include_directories(SFML::System INTERFACE ${SFML_INCLUDE_DIR})
        
        add_library(SFML::Window UNKNOWN IMPORTED)
        set_target_properties(SFML::Window PROPERTIES IMPORTED_LOCATION ${SFML_WINDOW_LIBRARY})
        target_include_directories(SFML::Window INTERFACE ${SFML_INCLUDE_DIR})
        
        add_library(SFML::Graphics UNKNOWN IMPORTED)
        set_target_properties(SFML::Graphics PROPERTIES IMPORTED_LOCATION ${SFML_GRAPHICS_LIBRARY})
        target_include_directories(SFML::Graphics INTERFACE ${SFML_INCLUDE_DIR})
    endif()
endif()
EOF
fi

echo "Installation de GLM pour $VCPKG_TARGET..."
./vcpkg/vcpkg install glm:$VCPKG_TARGET

echo ""
echo "=== État des installations ==="
./vcpkg/vcpkg list

echo ""
echo "✅ Correction terminée! Vous pouvez maintenant utiliser build-macos.sh"
