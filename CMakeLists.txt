cmake_minimum_required(VERSION 3.20)

# Configuration vcpkg pour macOS
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project(VCRasterizer VERSION 1.0.0)

# Configuration C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuration pour différents compilateurs
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Configuration pour Clang (macOS) et GCC (Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    
    # Support AVX2 pour SIMD (vérifier la compatibilité du processeur)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
endif()

# Trouver SFML
find_package(SFML 3.0.0 COMPONENTS Graphics Window System REQUIRED)

# GLM
find_package(glm CONFIG REQUIRED)

# Collecter tous les fichiers source
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.h"
    "include/*.h"
)

# Créer l'exécutable
add_executable(${PROJECT_NAME} ${SOURCES})

# Lier les bibliothèques
target_link_libraries(${PROJECT_NAME} 
    SFML::Graphics 
    SFML::Window 
    SFML::System
    glm::glm
)

# Sur macOS, lier les frameworks nécessaires
if(APPLE)
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    target_link_libraries(${PROJECT_NAME} 
        ${COCOA_LIBRARY} 
        ${IOKIT_LIBRARY} 
        ${COREVIDEO_LIBRARY}
    )
endif()

# Inclure les répertoires
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Configuration des répertoires de sortie
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin/Debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin/Release
)

# Copier les assets dans le répertoire de build
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets 
     DESTINATION ${CMAKE_BINARY_DIR}/bin)
