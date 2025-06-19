#pragma once

#include <vector>
#include "LogicGate.h"
#include "Grid.h"

class DiffLogicCA {
public:
    DiffLogicCA() = default;

    // Add components from loaded configuration
    void add_perception_kernel(const std::vector<LogicGate::Type>& gates);
    void add_update_layer(const std::vector<LogicGate::Type>& gates);

    // Run one step of the CA
    void step(Grid& grid);

private:
    // Helper functions
    std::vector<bool> get_neighborhood(const Grid& grid, int row, int col);
    bool apply_perception_kernel(const std::vector<bool>& inputs, const std::vector<LogicGate::Type>& kernel);
    bool apply_update_network(const std::vector<bool>& inputs);

    // Model components
    std::vector<std::vector<LogicGate::Type>> perception_kernels;
    std::vector<std::vector<LogicGate::Type>> update_network;
};
