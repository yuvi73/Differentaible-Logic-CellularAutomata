print("[DEBUG] This is the REAL diff_logic_ca.py")
import torch
import torch.nn as nn
from diff_logic_gates import DifferentiableLogicLayer

class PerceptionKernel(nn.Module):
    """
    A perception kernel that processes the neighborhood of a cell.
    Based on the paper's architecture: [8, 4, 2, 1] nodes.
    """
    def __init__(self, input_size=9):
        super().__init__()
        self.input_size = input_size
        self.layer1 = DifferentiableLogicLayer(8, input_size)
        self.layer2 = DifferentiableLogicLayer(4, 8)
        self.layer3 = DifferentiableLogicLayer(2, 4)
        self.layer4 = DifferentiableLogicLayer(1, 2)
    def forward(self, x, temperature=1.0):
        x = self.layer1(x, temperature)
        x = self.layer2(x, temperature)
        x = self.layer3(x, temperature)
        x = self.layer4(x, temperature)
        return x
    def get_gate_operations(self):
        operations = []
        operations.extend(self.layer1.get_gate_operations())
        operations.extend(self.layer2.get_gate_operations())
        operations.extend(self.layer3.get_gate_operations())
        operations.extend(self.layer4.get_gate_operations())
        return operations

class UpdateNetwork(nn.Module):
    """
    The update network that determines the new state of each cell.
    Based on the paper's architecture: 16 layers of 128 nodes, then [64, 32, 16, 8, 4, 2, 1].
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        layers = []
        current_size = input_size
        for _ in range(16):
            layers.append(DifferentiableLogicLayer(128, current_size))
            current_size = 128
        for size in [64, 32, 16, 8, 4, 2, 1]:
            layers.append(DifferentiableLogicLayer(size, current_size))
            current_size = size
        self.layers = nn.ModuleList(layers)
    def forward(self, x, temperature=1.0):
        for layer in self.layers:
            x = layer(x, temperature)
        return x
    def get_gate_operations(self):
        operations = []
        for layer in self.layers:
            operations.extend(layer.get_gate_operations())
        return operations

class DiffLogicCA(nn.Module):
    """
    Differentiable Logic Cellular Automata model.
    Implements the two-stage update mechanism: perception and update.
    """
    def __init__(self, num_channels=1, num_kernels=16):
        super().__init__()
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        # Create 16 perception kernels per channel
        self.perception_kernels = nn.ModuleList([
            PerceptionKernel() for _ in range(num_kernels * num_channels)
        ])
        update_input_size = num_channels + (num_channels * num_kernels)
        self.update_network = UpdateNetwork(update_input_size)
    def forward(self, grid, temperature=1.0):
        batch_size, height, width, channels = grid.shape
        grid_flat = grid.view(batch_size, -1)  # (batch_size, 9)
        perception_outputs = []
        for ch in range(channels):
            channel_input = grid_flat  # (batch_size, 9)
            kernel_outputs = []
            for k in range(self.num_kernels):
                kernel = self.perception_kernels[ch * self.num_kernels + k]
                output = kernel(channel_input, temperature)
                kernel_outputs.append(output)
            channel_perception = torch.cat(kernel_outputs, dim=1)
            perception_outputs.append(channel_perception)
        all_perception = torch.cat(perception_outputs, dim=1)
        update_input = torch.cat([grid_flat, all_perception], dim=1)
        update_output = self.update_network(update_input, temperature)
        updated_grid = update_output.view(batch_size, 1)
        return updated_grid
    def get_circuit_config(self):
        """
        Extract the learned logic circuit configuration as a serializable dictionary.
        """
        config = {
            "num_channels": self.num_channels,
            "num_kernels": self.num_kernels,
            "perception_kernels": [],
            "update_network": []
        }
        # Perception kernels
        for kernel in self.perception_kernels:
            kernel_ops = kernel.get_gate_operations()
            config["perception_kernels"].append(kernel_ops)
        # Update network
        for layer in self.update_network.layers:
            layer_ops = layer.get_gate_operations()
            config["update_network"].append(layer_ops)
        return config