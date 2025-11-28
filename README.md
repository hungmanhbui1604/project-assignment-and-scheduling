# Project Assignment and Scheduling Optimization

This repository contains two optimization algorithms for solving the project assignment and scheduling problem:

1. **Particle Swarm Optimization (PSO)** - [`pso.py`](pso.py)
2. **Ant Colony Optimization (ACO)** - [`aco.py`](aco.py)

Both algorithms read problem data from standard input and output optimal task assignments with start times.

## Problem Description

The project assignment and scheduling problem involves:
- **N tasks** with durations and precedence constraints
- **M teams** with availability times
- **Task-team assignments** with associated costs

The goal is to assign tasks to teams and schedule them to minimize:
1. Number of unscheduled tasks (primary objective)
2. Total project completion time (makespan)
3. Total assignment cost

## Input Format

The input file (e.g., [`input.txt`](input.txt)) follows this structure:

```
N Q                    # N: number of tasks, Q: number of precedence pairs
i j                    # Q lines: task i must finish before task j starts
d(1) d(2) ... d(N)    # Durations for each task
M                      # M: number of teams
s(1) s(2) ... s(M)    # Start times for each team
K                      # K: number of allowed task-team assignments
i j c                  # K lines: task i can be assigned to team j with cost c
```

### Example Input Structure

```
100 100                # 100 tasks, 100 precedence constraints
94 70                  # Task 94 must finish before task 70 starts
92 65                  # Task 92 must finish before task 65 starts
...                    # More precedence constraints
60 80 30 90 ...        # Durations for tasks 1-100
80                     # 80 teams
20 40 80 0 60 ...      # Start times for teams 1-80
1580                   # 1580 allowed assignments
1 4 90                 # Task 1 can be assigned to team 4 with cost 90
1 6 180                # Task 1 can be assigned to team 6 with cost 180
...                    # More allowed assignments
```

## Running the Algorithms

### Prerequisites

- Python 3.x installed on your system
- No additional dependencies required (uses only standard library)

### Method 1: Using Input Redirection (Recommended)

```bash
# Run PSO algorithm
python pso.py < input.txt

# Run ACO algorithm
python aco.py < input.txt
```

### Method 2: Piping Input

```bash
# Run PSO algorithm
cat input.txt | python pso.py

# Run ACO algorithm
cat input.txt | python aco.py
```

### Method 3: Interactive Input

```bash
# Run and paste input manually
python pso.py
# Then paste the input content and press Ctrl+D (Unix) or Ctrl+Z (Windows)
```

## Output Format

Both algorithms output results in the same format:

```
R                      # Number of scheduled tasks
i j u                  # R lines: task i assigned to team j starting at time u
```

### Example Output

```
100                    # All 100 tasks were scheduled
1 4 20                 # Task 1 assigned to team 4, starts at time 20
2 7 0                  # Task 2 assigned to team 7, starts at time 0
...                    # More task assignments
```

## Algorithm Parameters

### PSO Parameters (in [`pso.py`](pso.py))

- `SWARM_SIZE = 50`: Number of particles in the swarm
- `MAX_ITERATIONS = 150`: Maximum number of iterations
- `W = 0.75`: Inertia weight
- `C1 = 1.5`: Cognitive coefficient
- `C2 = 1.5`: Social coefficient
- `TIME_LIMIT = 270.0`: Maximum execution time in seconds

### ACO Parameters (in [`aco.py`](aco.py))

- `ALPHA = 1.0`: Pheromone influence
- `BETA = 2.0`: Heuristic influence
- `RHO = 0.1`: Evaporation rate
- `NUM_ANTS = 20`: Number of ants
- `ITERATIONS = 100`: Maximum number of iterations
- `Q = 1000.0`: Pheromone deposit factor
- `TIME_LIMIT = 270.0`: Maximum execution time in seconds

## Performance Comparison

Both algorithms implement the same problem formulation but use different optimization strategies:

- **PSO** uses a population-based approach with particles exploring the solution space
- **ACO** uses probabilistic construction based on pheromone trails and heuristic information

You can compare their performance by running both with the same input:

```bash
# Run both and save outputs
python pso.py < input.txt > pso_output.txt
python aco.py < input.txt > aco_output.txt

# Compare the number of scheduled tasks
head -n 1 pso_output.txt
head -n 1 aco_output.txt
```

## Customization

To modify algorithm parameters, edit the constant values at the top of each script:

- For PSO: Modify constants in [`pso.py`](pso.py) lines 6-11
- For ACO: Modify constants in [`aco.py`](aco.py) lines 5-11

## Time Limits

Both algorithms include a 270-second time limit (4.5 minutes) to ensure reasonable execution times. The algorithms will stop early if this limit is reached during optimization.

## Troubleshooting

### Common Issues

1. **"No such file or directory"**: Ensure you're in the correct directory and the input file exists
2. **Python not found**: Install Python 3.x or use `python3` instead of `python`
3. **Permission denied**: Make sure the Python scripts have execute permissions (`chmod +x pso.py aco.py`)

### Performance Tips

- For larger problems, consider increasing `TIME_LIMIT` to allow more optimization time
- Adjust population/swarm sizes based on your system's computational resources
- For faster results, reduce `MAX_ITERATIONS` or `ITERATIONS`

## File Structure

```
.
├── README.md           # This file
├── input.txt           # Example input file
├── pso.py              # Particle Swarm Optimization implementation
├── aco.py              # Ant Colony Optimization implementation
├── quick_tuner.py      # Hyperparameter tuning utility
├── calculate_metrics.py # Performance metrics calculator
└── problem/            # Directory with problem illustrations
    ├── problem1.png
    ├── problem2.png
    └── problem3.png
```

## Hyperparameter Tuning Utility

A comprehensive tuning utility [`quick_tuner.py`](quick_tuner.py) is provided to automatically find the best hyperparameters for both optimization algorithms:

### Features

- **Multi-algorithm support**: Tune ACO, PSO, or both algorithms simultaneously
- **Automated parameter generation**: Tests diverse parameter combinations
- **Performance comparison**: Ranks configurations by fitness score
- **Detailed results**: Shows scheduled tasks, makespan, cost, and execution time
- **Result persistence**: Saves all results to timestamped JSON files

### Usage

```bash
# Tune both algorithms with default 15 configurations each
python quick_tuner.py input.txt

# Tune only ACO algorithm
python quick_tuner.py input.txt aco

# Tune only PSO algorithm with 20 configurations
python quick_tuner.py input.txt pso 20

# Tune both with 25 configurations each
python quick_tuner.py input.txt both 25
```

### Parameter Ranges Tested

**ACO Parameters:**
- `alpha`: [0.5, 1.0, 1.5, 2.0] - Pheromone influence
- `beta`: [1.0, 2.0, 3.0, 4.0] - Heuristic influence
- `rho`: [0.05, 0.1, 0.2, 0.3] - Evaporation rate
- `num_ants`: [10, 20, 30, 40, 50] - Number of ants
- `iterations`: [50, 100, 150] - Maximum iterations
- `q`: [500, 1000, 1500, 2000] - Pheromone deposit factor

**PSO Parameters:**
- `swarm_size`: [20, 30, 50, 70, 100] - Number of particles
- `max_iterations`: [100, 150, 200] - Maximum iterations
- `w`: [0.4, 0.6, 0.75, 0.8, 0.9] - Inertia weight
- `c1`: [0.5, 1.0, 1.5, 2.0] - Cognitive coefficient
- `c2`: [0.5, 1.0, 1.5, 2.0] - Social coefficient

### Fitness Function

The tuner uses a composite fitness function (lower is better):
```
fitness = (unscheduled_tasks × 1,000,000) + (makespan × 1,000) + total_cost
```

This prioritizes:
1. **Maximizing scheduled tasks** (primary objective)
2. **Minimizing completion time** (secondary objective)
3. **Minimizing total cost** (tertiary objective)

### Example Output

```
Quick Hyperparameter Tuning
Input: input.txt
Algorithm: both
Configs per algorithm: 15
Problem size: 100 tasks

==================================================
TESTING ACO ALGORITHM
==================================================

ACO Config 1/15: {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1, 'num_ants': 20, 'iterations': 100, 'q': 1000}
  ✅ Scheduled: 95/100
     Makespan: 1250.0, Cost: 87500.0
     Fitness: 5200000.0, Time: 45.2s

...

==================================================
TOP 5 ACO CONFIGURATIONS
==================================================

1. Fitness: 5200000.0
   Config: {'alpha': 1.5, 'beta': 3.0, 'rho': 0.05, 'num_ants': 30, 'iterations': 150, 'q': 1500}
   Results: Scheduled 95/100, Makespan 1250.0, Cost 87500.0

...

==================================================
ALGORITHM COMPARISON
==================================================

Best ACO:
  Fitness: 5200000.0
  Scheduled: 95/100
  Config: {'alpha': 1.5, 'beta': 3.0, 'rho': 0.05, ...}

Best PSO:
  Fitness: 5300000.0
  Scheduled: 94/100
  Config: {'swarm_size': 50, 'max_iterations': 150, 'w': 0.75, ...}

🏆 ACO performs better on this dataset!

💾 Results saved to: tuning_results_1701234567.json
```

### Benefits

- **Find optimal parameters** without manual trial-and-error
- **Compare algorithms** objectively on your specific problem instance
- **Save time** by identifying the best configuration quickly
- **Reproducible results** with saved parameter sets and performance metrics

## Performance Metrics Calculator

A separate utility [`calculate_metrics.py`](calculate_metrics.py) is provided to analyze the performance of algorithm outputs:

### Usage

```bash
# First generate output from an algorithm
python aco.py < input.txt > aco_output.txt
# or
python pso.py < input.txt > pso_output.txt

# Then calculate metrics
python calculate_metrics.py input.txt aco_output.txt
# or
python calculate_metrics.py input.txt pso_output.txt
```

### Metrics Calculated

The utility calculates three key performance metrics:

1. **Number of tasks scheduled** - Count of successfully assigned tasks
2. **Completion time (makespan)** - Maximum finish time across all scheduled tasks
3. **Total cost** - Sum of costs for all task-team assignments

### Example Output

```
==================================================
SCHEDULING METRICS
==================================================
Number of tasks scheduled: 95
Completion time (makespan): 1250
Total cost: 87500
==================================================
```

This allows you to easily compare the performance of different algorithms on the same problem instance by running each algorithm and then analyzing their outputs with the metrics calculator.

## References

- Particle Swarm Optimization: Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization"
- Ant Colony Optimization: Dorigo, M., & Gambardella, L. M. (1997). "Ant colonies for the travelling salesman problem"