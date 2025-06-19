# Differentiable Logic Cellular Automata

A novel approach to learning cellular automata rules using differentiable logic gates. This project demonstrates learning Conway's Game of Life rules through a neural network-like architecture that uses interpretable logic operations instead of traditional neural network weights.

## Key Features

- **Differentiable Logic Gates**: Implementation of 16 basic logic operations as differentiable functions
- **Two-Stage Architecture**: 
  - Perception kernels for pattern recognition
  - Update network for rule application
- **Interpretable Results**: Learned rules can be implemented as actual logic circuits
- **Efficient C++ Implementation**: Fast execution of learned rules
- **PyTorch Training Framework**: Robust training pipeline for learning CA rules

## Project Structure

```
.
├── include/                 # C++ header files
│   ├── LogicGate.h         # Logic gate operations
│   ├── Grid.h              # CA grid management
│   ├── DiffLogicCA.h       # Main CA implementation
│   └── ModelLoader.h       # JSON configuration loader
├── src/                    # C++ source files
│   ├── main.cpp           # Simulation and visualization
│   ├── DiffLogicCA.cpp    # CA implementation
│   └── ModelLoader.cpp    # Model loading utilities
├── training/              # Python training framework
│   ├── diff_logic_gates.py    # Differentiable logic gates
│   ├── diff_logic_ca.py       # CA training implementation
│   ├── game_of_life_data.py   # Training data generation
│   └── train_game_of_life.py  # Training script
└── CMakeLists.txt        # Build configuration
```

## Technical Details

### Differentiable Logic Gates
- 16 basic operations (AND, OR, NOT, etc.)
- Continuous relaxation for gradient-based learning
- Learnable parameters to select optimal operations

### Architecture
1. **Perception Network**:
   - Multiple kernels process cell neighborhoods
   - [8, 4, 2, 1] node architecture per kernel
   
2. **Update Network**:
   - 16 layers of 128 nodes
   - Followed by [64, 32, 16, 8, 4, 2, 1] layers
   - Determines next cell state

## Building and Running

### Prerequisites
- CMake (>= 3.10)
- C++ compiler with C++17 support
- Python 3.8+ with PyTorch for training

### Training the Model
```bash
cd training
pip install -r requirements.txt
python train_game_of_life.py
```

### Building C++ Implementation
```bash
mkdir build && cd build
cmake ..
make
```

### Running the Simulation
```bash
./game_of_life
```

## Implementation Details

### Training Phase
- Uses PyTorch for differentiable logic operations
- Learns Game of Life rules through gradient descent
- Saves learned configuration to JSON

### C++ Implementation
- Efficient execution of learned rules
- Real-time visualization using Unicode characters
- Supports various initial patterns (blinker, glider)

## Novel Contributions

1. **Logic-Based Learning**: Uses interpretable logic operations instead of abstract neural network weights
2. **Bridge to Hardware**: Learned rules can be implemented directly in digital circuits
3. **Two-Stage Architecture**: Separates pattern recognition from rule application
4. **Continuous Relaxation**: Makes discrete logic operations differentiable

## Future Work

- Support for learning other cellular automata rules
- Hardware implementation of learned rules
- Interactive pattern creation
- Performance optimizations
- Extended pattern library

## License

This project is open source and available under the MIT License.
