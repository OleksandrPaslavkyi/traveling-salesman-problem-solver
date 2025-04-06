# Traveling Salesman Problem (TSP) Solver

## Overview
This project aims to solve the **Traveling Salesman Problem (TSP)** using three different algorithms:
- **Random Sampling**
- **Greedy Algorithm**
- **Genetic Algorithm**

The goal of the TSP is to find the shortest possible route that visits each city exactly once and returns to the origin city. In this project, we apply multiple methods to find the optimal or near-optimal solution for TSP.

## Methods Used

### 1. **Random Sampling**
The random sampling method generates a specified number of random routes and calculates the total distance for each route. The route with the shortest total distance is selected as the best solution.

- **Advantages**: Simple to implement, no need for any complex logic.
- **Disadvantages**: Inefficient for large datasets as it does not use any optimization strategy. It may take a long time to find a good solution.

### 2. **Greedy Algorithm**
The greedy algorithm starts with a randomly chosen city and then repeatedly selects the nearest unvisited city as the next city to visit. This process continues until all cities are visited.

- **Advantages**: Faster and more efficient than random sampling. It provides a decent solution for smaller datasets.
- **Disadvantages**: The greedy approach can get stuck in local optima, meaning that it does not always find the best solution.

### 3. **Genetic Algorithm**
The genetic algorithm (GA) is a more advanced optimization technique inspired by the process of natural selection. In the context of TSP, it works as follows:
1. **Initialization**: Create an initial population of random routes.
2. **Selection**: Select individuals based on fitness (shorter distance).
3. **Crossover**: Combine two parents to create new offspring by swapping segments of their routes.
4. **Mutation**: Randomly swap cities in a route with a given mutation rate to explore new solutions.
5. **Repeat**: Repeat this process for a set number of generations, selecting the best routes over time.

- **Advantages**: Can find high-quality solutions for large datasets and complex problems.
- **Disadvantages**: Computationally intensive and requires careful tuning of parameters (like mutation rate, population size, etc.).

## How to Run

1. **Install Dependencies**:
   Make sure you have Python 3.x installed. Then, install the necessary libraries by running:

   ```bash
   pip install numpy matplotlib
