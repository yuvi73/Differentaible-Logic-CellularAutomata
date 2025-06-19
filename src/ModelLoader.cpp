#include "ModelLoader.h"
#include <fstream>
#include <iostream>

void ModelLoader::load_model(const std::string& filename, DiffLogicCA& ca) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    nlohmann::json config;
    file >> config;

    // Load perception kernels
    load_perception_kernels(config, ca);

    // Load update network
    load_update_network(config, ca);

    std::cout << "Successfully loaded model from " << filename << std::endl;
}

void ModelLoader::load_perception_kernels(const nlohmann::json& config, DiffLogicCA& ca) {
    const auto& kernels = config["perception_kernels"];
    for (const auto& kernel_ops : kernels) {
        std::vector<LogicGate::Type> gates;
        for (const auto& op : kernel_ops) {
            gates.push_back(gate_index_to_type(op));
        }
        ca.add_perception_kernel(gates);
    }
}

void ModelLoader::load_update_network(const nlohmann::json& config, DiffLogicCA& ca) {
    const auto& layers = config["update_network"];
    for (const auto& layer_ops : layers) {
        std::vector<LogicGate::Type> gates;
        for (const auto& op : layer_ops) {
            gates.push_back(gate_index_to_type(op));
        }
        ca.add_update_layer(gates);
    }
}

LogicGate::Type ModelLoader::gate_index_to_type(int index) {
    switch (index) {
        case 0: return LogicGate::Type::ZERO;
        case 1: return LogicGate::Type::AND;
        case 2: return LogicGate::Type::A_NOT_B;
        case 3: return LogicGate::Type::A;
        case 4: return LogicGate::Type::B_NOT_A;
        case 5: return LogicGate::Type::B;
        case 6: return LogicGate::Type::XOR;
        case 7: return LogicGate::Type::OR;
        case 8: return LogicGate::Type::NOR;
        case 9: return LogicGate::Type::XNOR;
        case 10: return LogicGate::Type::NOT_B;
        case 11: return LogicGate::Type::A_OR_NOT_B;
        case 12: return LogicGate::Type::NOT_A;
        case 13: return LogicGate::Type::NOT_A_OR_B;
        case 14: return LogicGate::Type::NAND;
        case 15: return LogicGate::Type::ONE;
        default: throw std::runtime_error("Invalid gate index: " + std::to_string(index));
    }
} 