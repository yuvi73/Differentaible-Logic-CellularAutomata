#pragma once
#include "LogicGate.h"
#include "LogicCircuit.h"
#include <string>
#include <vector>
#include "json.hpp"
#include "DiffLogicCA.h"

// Converts a string to LogicGate::Type
LogicGate::Type gate_type_from_string(const std::string& s);

// Loads a LogicCircuit from a JSON object
LogicCircuit load_circuit(const nlohmann::json& j);

// Loads the model from a JSON file
void load_model(const std::string& filename,
                std::vector<LogicCircuit>& perception_kernels,
                LogicCircuit& update_network);

class ModelLoader {
public:
    static void load_model(const std::string& filename, DiffLogicCA& ca);

private:
    static void load_perception_kernels(const nlohmann::json& config, DiffLogicCA& ca);
    static void load_update_network(const nlohmann::json& config, DiffLogicCA& ca);
    static LogicGate::Type gate_index_to_type(int index);
}; 