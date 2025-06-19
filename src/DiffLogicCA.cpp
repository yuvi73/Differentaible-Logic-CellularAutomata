#include "DiffLogicCA.h"
#include <stdexcept>

void DiffLogicCA::add_perception_kernel(const std::vector<LogicGate::Type>& gates) {
    perception_kernels.push_back(gates);
}

void DiffLogicCA::add_update_layer(const std::vector<LogicGate::Type>& gates) {
    update_network.push_back(gates);
}

std::vector<bool> DiffLogicCA::get_neighborhood(const Grid& grid, int row, int col) {
    std::vector<bool> neighborhood;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int x = (col + dx + grid.width()) % grid.width();
            int y = (row + dy + grid.height()) % grid.height();
            neighborhood.push_back(grid.get(x, y));
        }
    }
    return neighborhood;
}

bool DiffLogicCA::apply_perception_kernel(const std::vector<bool>& inputs, 
                                        const std::vector<LogicGate::Type>& kernel) {
    if (inputs.size() < 2) {
        throw std::runtime_error("Not enough inputs for perception kernel");
    }

    std::vector<bool> current_layer = inputs;
    std::vector<bool> next_layer;

    for (size_t i = 0; i < kernel.size(); ++i) {
        next_layer.clear();
        for (size_t j = 0; j < current_layer.size() - 1; j += 2) {
            bool result = LogicGate::apply(kernel[i], current_layer[j], current_layer[j + 1]);
            next_layer.push_back(result);
        }
        if (current_layer.size() % 2 == 1) {
            next_layer.push_back(current_layer.back());
        }
        current_layer = next_layer;
    }

    return current_layer[0];
}

bool DiffLogicCA::apply_update_network(const std::vector<bool>& inputs) {
    std::vector<bool> current_layer = inputs;
    std::vector<bool> next_layer;

    for (const auto& layer : update_network) {
        next_layer.clear();
        for (size_t i = 0; i < layer.size(); ++i) {
            size_t input1_idx = i % current_layer.size();
            size_t input2_idx = (i + 1) % current_layer.size();
            bool result = LogicGate::apply(layer[i], 
                                         current_layer[input1_idx], 
                                         current_layer[input2_idx]);
            next_layer.push_back(result);
        }
        current_layer = next_layer;
    }

    return current_layer[0];
}

void DiffLogicCA::step(Grid& grid) {
    Grid next_grid(grid.width(), grid.height());

    for (int y = 0; y < grid.height(); ++y) {
        for (int x = 0; x < grid.width(); ++x) {
            // Get 3x3 neighborhood
            auto neighborhood = get_neighborhood(grid, y, x);

            // Apply perception kernels
            std::vector<bool> perception_outputs;
            for (const auto& kernel : perception_kernels) {
                bool output = apply_perception_kernel(neighborhood, kernel);
                perception_outputs.push_back(output);
            }

            // Combine current state with perception outputs
            std::vector<bool> update_inputs = {grid.get(x, y)};
            update_inputs.insert(update_inputs.end(), 
                               perception_outputs.begin(), 
                               perception_outputs.end());

            // Apply update network
            bool new_state = apply_update_network(update_inputs);
            next_grid.set(x, y, new_state);
        }
    }

    // Update the grid
    grid = next_grid;
} 