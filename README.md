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
- `W = 0.729`: Inertia weight
- `C1 = 1.49445`: Cognitive coefficient
- `C2 = 1.49445`: Social coefficient
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
└── problem/            # Directory with problem illustrations
    ├── problem1.png
    ├── problem2.png
    └── problem3.png
```

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