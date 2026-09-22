cmake_minimum_required(VERSION 3.10)
project(LoopDetectorModule)

# AGGIUNTA CHIAVE: Compila in modalità Debug
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Trova le nostre dipendenze ---
add_subdirectory(extern/pybind11)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)

# --- Crea il nostro modulo Python come libreria condivisa ---
add_library(loop_detector SHARED
    cpp_src/bindings.cpp
    cpp_src/Scancontext.cpp
)

# --- Specifica i percorsi degli "header" per il nostro modulo ---
target_include_directories(loop_detector PRIVATE
    cpp_src/                  # I nostri file .h (types.h, etc.)
    extern/pybind11/include/  # Il percorso cruciale per pybind11
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

# --- Collega le librerie ---
target_link_libraries(loop_detector PRIVATE
    pybind11::module      # Collega i componenti base di pybind11
    pybind11::embed       # Collega i componenti per l'integrazione (necessario)
    ${PCL_LIBRARIES}
)

# --- Aggiungi un prefisso standard per i moduli Python ---
set_target_properties(loop_detector PROPERTIES PREFIX "")