# DiffLogic Cellular Automata (CA) - C++ Inference

This project implements the inference (deployment) phase of Differentiable Logic Cellular Automata (DiffLogic CA) in C++. It is designed for high efficiency and suitability for resource-constrained environments.

## Features
- Modular logic gate and circuit representation
- Two-stage update mechanism (perception + update)
- Easily extensible for any trained DiffLogic CA model

## Directory Structure
- `src/` - Source files
- `include/` - Header files
- `CMakeLists.txt` - Build configuration

## Build Instructions

This project uses [CMake](https://cmake.org/) for building.

```
mkdir build
cd build
cmake ..
make
./DiffLogicCA
```

## Usage
- Edit `src/main.cpp` to define your CA grid, perception kernels, and update network.
- Extend the code to load trained model parameters from a config file if needed.

## Requirements
- C++17 or later
- CMake 3.10+

## License
MIT
