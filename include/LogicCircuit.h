#pragma once
#include "LogicGate.h"
#include <vector>

struct LogicGateNode {
    LogicGate::Type gate_type;
    int input1_idx; // Index in previous layer
    int input2_idx; // Index in previous layer
};

struct CircuitLayer {
    std::vector<LogicGateNode> nodes;
};

struct LogicCircuit {
    std::vector<CircuitLayer> layers;

    // Forward pass through the circuit
    std::vector<bool> forward(const std::vector<bool>& input) const {
        std::vector<std::vector<bool>> activations;
        activations.push_back(input);
        for (const auto& layer : layers) {
            std::vector<bool> out;
            for (const auto& node : layer.nodes) {
                bool a = activations.back()[node.input1_idx];
                bool b = activations.back()[node.input2_idx];
                out.push_back(LogicGate::apply(node.gate_type, a, b));
            }
            activations.push_back(out);
        }
        return activations.back();
    }
};
