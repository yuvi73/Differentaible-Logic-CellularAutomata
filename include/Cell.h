#pragma once
#include <vector>

struct Cell {
    std::vector<bool> state;      // "Gray register" (current state)
    std::vector<bool> perception; // "Orange register" (perception output)
};
