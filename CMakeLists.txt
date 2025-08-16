cmake_minimum_required(VERSION 3.20)

# Détection de l'architecture avant la configuration vcpkg
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(VCPKG_TARGET_TRIPLET "arm64-osx" CACHE STRING "")
    set(ARCH_ARM64 TRUE)
    message(STATUS "Detected ARM64 architecture, using arm64-osx triplet")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(VCPKG_TARGET_TRIPLET "x64-osx" CACHE STRING "")
    set(ARCH_X64 TRUE)
    message(STATUS "Detected x64 architecture, using x64-osx triplet")
endif()

# Configuration vcpkg avec le bon triplet
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
    
    # Support AVX2 sur x64 Windows
    if(ARCH_X64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Configuration pour Clang (macOS) et GCC (Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    
    # Support SIMD selon l'architecture
    if(ARCH_ARM64)
        # Support ARM NEON
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
        add_compile_definitions(SIMD_ARM_NEON)
        message(STATUS "Enabling ARM NEON support")
        
        # Vérifier si le compilateur supporte les intrinsics NEON
        include(CheckCXXSourceCompiles)
        check_cxx_source_compiles("
            #include <arm_neon.h>
            int main() { 
                float32x4_t v = vdupq_n_f32(1.0f); 
                return 0; 
            }" HAVE_NEON)
        
        if(NOT HAVE_NEON)
            message(WARNING "ARM NEON intrinsics not supported by compiler")
        endif()
        
    elseif(ARCH_X64)
        # Support AVX2 sur x64
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        add_compile_definitions(SIMD_AVX2)
        message(STATUS "Enabling AVX2 support")
    endif()
endif()

# Trouver SFML - essayer différentes versions et configurations
find_package(SFML 3.0.0 COMPONENTS Graphics Window System)
if(NOT SFML_FOUND)
    message(STATUS "SFML 3.0 not found via vcpkg, trying Homebrew...")
    # Fallback pour SFML via Homebrew
    if(APPLE)
        set(CMAKE_PREFIX_PATH "/opt/homebrew;/usr/local" ${CMAKE_PREFIX_PATH})
        find_package(SFML 2.6.0 COMPONENTS graphics window system)
        if(SFML_FOUND)
            # Créer des alias pour la compatibilité
            if(NOT TARGET SFML::Graphics)
                add_library(SFML::Graphics ALIAS sfml-graphics)
                add_library(SFML::Window ALIAS sfml-window) 
                add_library(SFML::System ALIAS sfml-system)
            endif()
        endif()
    endif()
endif()

if(NOT SFML_FOUND)
    message(FATAL_ERROR "SFML not found. Please install via vcpkg or Homebrew")
endif()

# GLM
find_package(glm CONFIG)
if(NOT glm_FOUND)
    message(STATUS "GLM not found via vcpkg, trying system installation...")
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GLM glm)
        if(GLM_FOUND)
            add_library(glm::glm INTERFACE IMPORTED)
            target_include_directories(glm::glm INTERFACE ${GLM_INCLUDE_DIRS})
        endif()
    endif()
endif()

if(NOT TARGET glm::glm)
    message(FATAL_ERROR "GLM not found. Please install via vcpkg or package manager")
endif()

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
    
    # Configuration spéciale pour Apple Silicon
    if(ARCH_ARM64)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            OSX_ARCHITECTURES "arm64"
        )
    endif()
endif()

# Inclure les répertoires
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Définir les macros selon l'architecture
if(ARCH_ARM64)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SIMD_ARM_NEON)
elseif(ARCH_X64)
    target_compile_definitions(${PROJECT_NAME} PRIVATE SIMD_AVX2)
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

# Afficher les informations de compilation
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
if(ARCH_ARM64)
    message(STATUS "SIMD Support: ARM NEON")
    message(STATUS "vcpkg Triplet: arm64-osx")
elseif(ARCH_X64)
    message(STATUS "SIMD Support: AVX2")
    message(STATUS "vcpkg Triplet: x64-osx")
else()
    message(STATUS "SIMD Support: Scalar fallback")
endif()
