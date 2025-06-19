cmake_minimum_required(VERSION 3.10)
project(DiffLogicCA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cpp
    src/DiffLogicCA.cpp
    src/ModelLoader.cpp
)

# Create executable
add_executable(game_of_life ${SOURCES})

# Copy configuration file to build directory
configure_file(${PROJECT_SOURCE_DIR}/learned_game_of_life_config.json
              ${CMAKE_CURRENT_BINARY_DIR}/learned_game_of_life_config.json
              COPYONLY) 