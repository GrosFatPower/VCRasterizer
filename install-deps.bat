@echo off
echo Installation des dépendances...

REM Installer vcpkg si nécessaire
if not exist vcpkg (
    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    .\bootstrap-vcpkg.bat
    cd ..
)

REM Installer SFML
vcpkg\vcpkg install sfml:x64-windows

echo Dépendances installées !
pause
