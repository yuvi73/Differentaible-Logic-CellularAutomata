import torch
import numpy as np
from itertools import product

def game_of_life_rules(grid):
    """
    Apply Conway's Game of Life rules to a 3x3 grid.
    
    Args:
        grid: 3x3 numpy array or tensor
    
    Returns:
        Next state of the grid
    """
    if isinstance(grid, torch.Tensor):
        grid = grid.numpy()
    
    # Pad the grid with zeros for boundary conditions
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    next_state = np.zeros_like(grid)
    
    for i in range(3):
        for j in range(3):
            # Count live neighbors (including diagonals)
            neighborhood = padded[i:i+3, j:j+3]
            live_neighbors = np.sum(neighborhood) - grid[i, j]
            
            # Apply Game of Life rules
            if grid[i, j] == 1:  # Live cell
                if live_neighbors == 2 or live_neighbors == 3:
                    next_state[i, j] = 1  # Survives
                else:
                    next_state[i, j] = 0  # Dies
            else:  # Dead cell
                if live_neighbors == 3:
                    next_state[i, j] = 1  # Birth
                else:
                    next_state[i, j] = 0  # Stays dead
    
    return next_state

def generate_all_3x3_configurations():
    """
    Generate all possible 3x3 binary grid configurations.
    There are 2^9 = 512 possible configurations.
    
    Returns:
        List of all possible 3x3 configurations
    """
    configurations = []
    
    # Generate all possible binary combinations for 9 cells
    for config in product([0, 1], repeat=9):
        # Reshape to 3x3
        grid = np.array(config).reshape(3, 3)
        configurations.append(grid)
    
    return configurations

def create_training_dataset():
    """
    Create the complete training dataset for Game of Life.
    
    Returns:
        input_grids: Tensor of shape (512, 3, 3, 1)
        target_grids: Tensor of shape (512, 3, 3, 1)
    """
    print("Generating all 3x3 Game of Life configurations...")
    
    # Generate all possible configurations
    all_configs = generate_all_3x3_configurations()
    
    input_grids = []
    target_grids = []
    
    for config in all_configs:
        # Apply Game of Life rules to get the next state
        next_state = game_of_life_rules(config)
        
        # Add channel dimension
        input_grids.append(config.reshape(3, 3, 1))
        target_grids.append(next_state.reshape(3, 3, 1))
    
    # Convert to tensors
    input_tensor = torch.tensor(input_grids, dtype=torch.float32)
    target_tensor = torch.tensor(target_grids, dtype=torch.float32)
    
    print(f"Created training dataset with {len(input_grids)} samples")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    
    return input_tensor, target_tensor

def visualize_sample(input_grid, target_grid, sample_idx=0):
    """
    Visualize a sample from the training dataset.
    
    Args:
        input_grid: Input tensor
        target_grid: Target tensor
        sample_idx: Index of sample to visualize
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Input grid
    ax1.imshow(input_grid[sample_idx, :, :, 0], cmap='binary', vmin=0, vmax=1)
    ax1.set_title(f'Input Grid (Sample {sample_idx})')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Target grid
    ax2.imshow(target_grid[sample_idx, :, :, 0], cmap='binary', vmin=0, vmax=1)
    ax2.set_title(f'Target Grid (Sample {sample_idx})')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def test_game_of_life_rules():
    """
    Test the Game of Life rules with some known patterns.
    """
    print("Testing Game of Life rules...")
    
    # Test 1: Still life (block)
    block = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    next_block = game_of_life_rules(block)
    print("Block pattern:")
    print(block)
    print("Next state:")
    print(next_block)
    print("Should remain unchanged:", np.array_equal(block, next_block))
    print()
    
    # Test 2: Blinker (oscillator)
    blinker = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ])
    next_blinker = game_of_life_rules(blinker)
    print("Blinker pattern:")
    print(blinker)
    print("Next state:")
    print(next_blinker)
    print()

if __name__ == "__main__":
    # Test the rules
    test_game_of_life_rules()
    
    # Generate training data
    input_data, target_data = create_training_dataset()
    
    # Save the data
    torch.save(input_data, 'training_data_input.pt')
    torch.save(target_data, 'training_data_target.pt')
    
    print("Training data saved to training_data_input.pt and training_data_target.pt") 