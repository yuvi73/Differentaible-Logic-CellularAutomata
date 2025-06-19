#include <iostream>
#include <chrono>
#include <thread>
#include "DiffLogicCA.h"
#include "ModelLoader.h"

// Function to print the grid
void print_grid(const Grid& grid) {
    for (int y = 0; y < grid.height(); ++y) {
        for (int x = 0; x < grid.width(); ++x) {
            std::cout << (grid.get(x, y) ? "■" : "□") << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    try {
        // Create CA with learned configuration
        DiffLogicCA ca;
        ModelLoader::load_model("learned_game_of_life_config.json", ca);

        // Create a grid (e.g., 20x20)
        Grid grid(20, 20);

        // Initialize with some patterns
        // Example: Blinker
        std::vector<std::vector<bool>> blinker = {
            {0, 1, 0},
            {0, 1, 0},
            {0, 1, 0}
        };
        grid.set_pattern(blinker, 5, 5);

        // Example: Glider
        std::vector<std::vector<bool>> glider = {
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 1}
        };
        grid.set_pattern(glider, 10, 10);

        // Run simulation
        std::cout << "Game of Life Simulation (Press Ctrl+C to stop)\n\n";
        while (true) {
            print_grid(grid);
            ca.step(grid);
            
            // Sleep for visualization
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Clear screen (works on Unix-like systems)
            std::cout << "\033[2J\033[H";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 