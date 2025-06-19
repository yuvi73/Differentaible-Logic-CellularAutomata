import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from diff_logic_ca import DiffLogicCA
from game_of_life_data import create_training_dataset, test_game_of_life_rules

def train_model(model, train_loader, num_epochs=1000, learning_rate=0.001, temperature=1.0):
    """
    Train the DiffLogic CA model.
    
    Args:
        model: DiffLogic CA model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        temperature: Temperature for softmax during training
    
    Returns:
        List of losses for plotting
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    print(f"Training on {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, temperature=temperature)
            
            # Compute loss
            center_targets = targets[:, 1, 1, 0].unsqueeze(1)  # shape: (batch_size, 1)
            loss = criterion(outputs, center_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained DiffLogic CA model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Average loss on test set
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs, temperature=0.1)  # Low temperature for sharp decisions
            center_targets = targets[:, 1, 1, 0].unsqueeze(1)  # shape: (batch_size, 1)
            loss = criterion(outputs, center_targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    return avg_loss

def plot_training_progress(losses):
    """
    Plot the training loss over time.
    
    Args:
        losses: List of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

def save_circuit_config(config, filename='learned_game_of_life_config.json'):
    """
    Save the learned circuit configuration to a JSON file.
    
    Args:
        config: Circuit configuration dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Circuit configuration saved to {filename}")

def main():
    """
    Main training function.
    """
    print("=== DiffLogic CA Game of Life Training ===")
    
    # Test Game of Life rules
    test_game_of_life_rules()
    
    # Create training dataset
    print("\n=== Creating Training Dataset ===")
    input_data, target_data = create_training_dataset()
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(input_data, target_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    print("\n=== Creating Model ===")
    model = DiffLogicCA(num_channels=1, num_kernels=16)
    
    # Train model
    print("\n=== Training Model ===")
    losses = train_model(model, train_loader, num_epochs=500, learning_rate=0.001)
    
    # Plot training progress
    plot_training_progress(losses)
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loss = evaluate_model(model, train_loader, device)
    print(f"Final test loss: {test_loss:.6f}")
    
    # Get learned circuit configuration
    print("\n=== Extracting Learned Circuit ===")
    circuit_config = model.get_circuit_config()
    
    # Save configuration
    save_circuit_config(circuit_config)
    
    print("\n=== Training Complete ===")
    print("You can now use the learned circuit configuration in your C++ DiffLogic CA!")

if __name__ == "__main__":
    main() 