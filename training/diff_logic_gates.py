print("[DEBUG] This is the REAL diff_logic_gates.py")
import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableLogicGate(nn.Module):
    """
    A differentiable logic gate that can learn which operation to perform.
    Based on the paper's continuous relaxations of logic operations.
    """
    def __init__(self, num_operations=16):
        super().__init__()
        self.num_operations = num_operations
        self.gate_logits = nn.Parameter(torch.zeros(num_operations))
        self.gate_logits.data[3] = 2.0  # A gate
        self.gate_logits.data[5] = 2.0  # B gate
    def forward(self, a, b, temperature=1.0):
        gate_probs = F.softmax(self.gate_logits / temperature, dim=0)
        operations = self._compute_continuous_operations(a, b)
        output = torch.sum(gate_probs.unsqueeze(0) * operations, dim=1)
        return output
    def _compute_continuous_operations(self, a, b):
        a = torch.clamp(a, 0, 1)
        b = torch.clamp(b, 0, 1)
        operations = torch.stack([
            torch.zeros_like(a),
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - b + a * b,
            1 - a,
            1 - a + a * b,
            1 - a * b,
            torch.ones_like(a)
        ], dim=1)
        return operations
    def get_most_probable_gate(self):
        gate_probs = F.softmax(self.gate_logits, dim=0)
        return torch.argmax(gate_probs).item()
    def get_gate_probabilities(self):
        return F.softmax(self.gate_logits, dim=0)

class DifferentiableLogicLayer(nn.Module):
    """
    A layer of differentiable logic gates.
    """
    def __init__(self, num_gates, input_size):
        print(f"[DEBUG] Creating DifferentiableLogicLayer: num_gates={num_gates}, input_size={input_size}")
        super().__init__()
        if input_size == 1 and num_gates > 1:
            raise ValueError("Cannot create a logic layer with input_size=1 and num_gates>1")
        self.num_gates = num_gates
        self.input_size = input_size
        self.gates = nn.ModuleList([DifferentiableLogicGate() for _ in range(num_gates)])
        self.connections = self._create_connections()
    def _create_connections(self):
        connections = []
        for i in range(self.num_gates):
            input1 = i % self.input_size
            input2 = (i + 1) % self.input_size if self.input_size > 1 else 0
            connections.append((input1, input2))
        return connections
    def forward(self, x, temperature=1.0):
        outputs = []
        for i, gate in enumerate(self.gates):
            input1_idx, input2_idx = self.connections[i]
            a = x[:, input1_idx]
            if input2_idx >= x.shape[1]:
                b = x[:, input1_idx]
            else:
                b = x[:, input2_idx]
            output = gate(a, b, temperature)
            outputs.append(output)
        return torch.stack(outputs, dim=1)
    def get_gate_operations(self):
        return [gate.get_most_probable_gate() for gate in self.gates]