# DiffLogic CA Training System

This directory contains the Python training implementation for learning Conway's Game of Life rules using Differentiable Logic Cellular Automata.

## Overview

The training system follows the approach described in the paper "Differentiable Logic Cellular Automata" and learns to implement Conway's Game of Life using differentiable logic gates.

## Files

- `diff_logic_gates.py`: Implementation of differentiable logic gates with continuous relaxations
- `diff_logic_ca.py`: DiffLogic CA model with perception and update networks
- `game_of_life_data.py`: Training data generation (all 512 possible 3x3 configurations)
- `train_game_of_life.py`: Main training script
- `requirements.txt`: Python dependencies

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training:**
   ```bash
   python train_game_of_life.py
   ```

## Training Process

1. **Data Generation**: Creates all 512 possible 3x3 grid configurations and their expected next states according to Game of Life rules
2. **Model Training**: Trains the DiffLogic CA model using differentiable logic gates
3. **Circuit Extraction**: Extracts the learned logic circuit configuration
4. **Output**: Saves the learned circuit as `learned_game_of_life_config.json`

## Model Architecture

- **Perception Kernels**: 16 kernels per channel, each with [8, 4, 2, 1] node structure
- **Update Network**: 23 layers: 16 layers of 128 nodes, then [64, 32, 16, 8, 4, 2, 1]
- **Training**: Uses MSE loss and Adam optimizer

## Output

The training produces:
- `learned_game_of_life_config.json`: Circuit configuration for your C++ deployment
- `training_loss.png`: Plot of training progress

## Using the Learned Model

1. Copy `learned_game_of_life_config.json` to your C++ project root
2. Update your C++ main.cpp to use this file:
   ```cpp
   load_model("learned_game_of_life_config.json", perception_kernels, update_network);
   ```
3. Build and run your C++ DiffLogic CA with the learned Game of Life rules!

## Expected Results

The trained model should learn to implement Conway's Game of Life rules, producing:
- Stable patterns (blocks, beehives)
- Oscillators (blinkers, toads)
- Moving patterns (gliders, spaceships)

## Notes

- Training may take several minutes depending on your hardware
- The model uses continuous relaxations during training but outputs discrete logic circuits
- The learned circuit will be much more complex than the simple examples in your original config files 