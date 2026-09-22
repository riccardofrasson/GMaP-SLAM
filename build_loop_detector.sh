#!/bin/bash
# Script to build the C++ Loop Detector module (loop_detector.so) using CMake

cd loop_closure
rm -rf build
mkdir build
cd build

# Generate Makefiles using CMake
cmake ..

# Compile the code (using 4 CPU cores to speed it up)
make -j4

# Copy the generated .so file to the Python directory so it can be imported
cp loop_detector*.so ../python_src/

echo "Build complete! The loop_detector.so module is now in loop_closure/python_src/"
