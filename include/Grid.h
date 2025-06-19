#pragma once

#include <vector>
#include <stdexcept>

class Grid {
public:
    Grid(int width, int height) 
        : width_(width), height_(height), cells_(width * height, false) {}

    bool get(int x, int y) const {
        if (!is_valid_position(x, y)) {
            throw std::out_of_range("Invalid grid position");
        }
        return cells_[y * width_ + x];
    }

    void set(int x, int y, bool value) {
        if (!is_valid_position(x, y)) {
            throw std::out_of_range("Invalid grid position");
        }
        cells_[y * width_ + x] = value;
    }

    int width() const { return width_; }
    int height() const { return height_; }

    // Initialize with a pattern
    void set_pattern(const std::vector<std::vector<bool>>& pattern, int start_x, int start_y) {
        for (size_t y = 0; y < pattern.size(); ++y) {
            for (size_t x = 0; x < pattern[y].size(); ++x) {
                int grid_x = (start_x + x) % width_;
                int grid_y = (start_y + y) % height_;
                set(grid_x, grid_y, pattern[y][x]);
            }
        }
    }

private:
    bool is_valid_position(int x, int y) const {
        return x >= 0 && x < width_ && y >= 0 && y < height_;
    }

    int width_;
    int height_;
    std::vector<bool> cells_;
};
