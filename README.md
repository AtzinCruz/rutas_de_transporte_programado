# Transportation Route Optimization with Metaheuristic Algorithms

This project addresses the problem of optimizing transportation routes using three metaheuristic algorithms: **Tabu Search**, **Basic Random Search (BAS)**, and **Localized Random Search (BAL)**. The goal is to minimize travel costs and times while meeting operational constraints, such as vehicle capacity.

## Problem Description

The primary problem is to efficiently organize transportation routes for a set of stops and destinations under the following constraints:
- **Maximum Capacity:** Buses cannot carry more than 20 people per trip.
- **Single Bus per Trip:** Each trip must be performed by a single bus.

Two main scenarios are considered for optimization:
1. Optimization between stops.
2. Optimization from stops to a destination plant.

## Metaheuristic Algorithms Used

### 1. Tabu Search
Tabu Search is a heuristic search method designed to explore the solution space beyond local optima by using an adaptive memory mechanism to avoid revisiting previously explored solutions. It was applied to minimize the total distance traveled by the bus, focusing on ensuring an exhaustive exploration of the solution space.

### 2. Basic Random Search (BAS)
BAS is a simple method based on generating random solutions and selecting the best among them. This approach is useful for providing an initial reference point or a quick solution that can be refined later.

### 3. Localized Random Search (BAL)
BAL improves the random search approach by restricting the search area to a neighborhood near the best-known solution, making the algorithm more efficient by focusing on promising regions of the solution space.

## Implementation Details

The implementation is divided into the following steps:
- **Preprocessing:** Prepare the data, including stops and destinations.
- **Initial Solution Generation:** Generate an initial feasible solution for optimization.
- **Neighbor Generation:** Generate neighboring solutions for each iteration.
- **Cost Calculation:** Calculate the cost of each route based on distance and other factors.
- **Iterative Improvement:** Each algorithm iteratively improves the current solution.

## Results
The experiments conducted demonstrated the following:
- **Tabu Search**: Achieved the most accurate solutions but required significantly more computation time.
- **BAS and BAL**: Were faster but less precise compared to Tabu Search.

## Conclusion
The choice of algorithm depends on the specific requirements of the problem:
- **Tabu Search** is ideal when accuracy is a priority, and there is enough computational capacity.
- **BAS and BAL** are more effective when computation time is limited, and approximate solutions are acceptable.

## Requirements
To run this project, the following dependencies are needed:
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`

Install the dependencies using:
```sh
pip install -r requirements.txt
```

## Running the Code
To run the optimization scripts, use the following commands:
```sh
python BAL.py
python BAS.py
python busqueda.py
```
Each script corresponds to the respective algorithm implementation.

## Authors
- Diego Issac Cano Vizcaino
- Atzin Eduardo Cruz Briones
- Max Georges Sainte Guzmán

## Acknowledgments
This project was developed as part of the **Optimization and Metaheuristics** course, supervised by Dr. Jonás Velasco Álvarez.
