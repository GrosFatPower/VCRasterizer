cmake_minimum_required(VERSION 3.20)

# Chemin vers vcpkg (ajustez selon votre installation)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(VCRasterizer VERSION 1.0.0)

# Configuration C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration pour Visual Studio
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    # Optimisations pour Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    # Debug info pour Debug
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
endif()

# Trouver SFML (nouvelle syntaxe pour SFML 2.6+)
find_package(SFML 3.0.0 COMPONENTS Graphics Window System REQUIRED)

# Si SFML n'est pas trouvé automatiquement, vous pouvez spécifier le chemin
# set(SFML_DIR "C:/SFML-2.5.1/lib/cmake/SFML")

# Collecter tous les fichiers source
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.h"
)

# Créer l'exécutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Lier les bibliothèques SFML (nouvelle syntaxe)
target_link_libraries(${PROJECT_NAME} 
    SFML::Graphics 
    SFML::Window 
    SFML::System
)

# Inclure les répertoires
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Copier les DLL SFML pour Windows (Debug et Release)
if(WIN32 AND MSVC)
    # Les DLL sont maintenant gérées automatiquement par vcpkg
    # Pas besoin de commandes manuelles avec SFML::targets
endif()

# Configuration des répertoires de sortie
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Copier les assets dans le répertoire de build
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets 
     DESTINATION ${CMAKE_BINARY_DIR}/bin)
