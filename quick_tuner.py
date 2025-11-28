#!/usr/bin/env python3
import sys
import time
import json
import random
from typing import Dict, List, Any

# Import both algorithms
sys.path.append('.')
from aco import ACO_Scheduler as ACO
from pso import solve_pso, convert_to_project_data

def parse_input_file(filename: str):
    """Parse input file and return data for both algorithms"""
    with open(filename, 'r') as f:
        input_data = f.read().split()

    iterator = iter(input_data)

    # Parse input
    N = int(next(iterator))
    Q = int(next(iterator))

    precedence = {i: [] for i in range(1, N + 1)}
    for _ in range(Q):
        i = int(next(iterator))
        j = int(next(iterator))
        precedence[j].append(i)

    durations = {}
    for i in range(1, N + 1):
        durations[i] = int(next(iterator))

    M = int(next(iterator))

    team_starts = []
    for _ in range(M):
        team_starts.append(int(next(iterator)))

    K_val = int(next(iterator))

    allowed_assignments = {}
    for _ in range(K_val):
        task_i = int(next(iterator))
        team_j = int(next(iterator))
        cost = int(next(iterator))
        allowed_assignments[(task_i, team_j)] = cost

    return (N, M, durations, precedence, team_starts, allowed_assignments)

def calculate_fitness(scheduled, makespan, cost, total_tasks):
    """Calculate fitness score (lower is better)"""
    unscheduled_penalty = (total_tasks - scheduled) * 1000000
    time_penalty = makespan * 1000
    cost_penalty = cost
    return unscheduled_penalty + time_penalty + cost_penalty

def test_single_aco_config(input_data, config):
    """Test a single ACO configuration"""
    N, M, durations, precedence, team_starts, allowed_assignments = input_data

    try:
        aco = ACO(
            N=N, M=M, durations=durations, precedence=precedence,
            team_starts=team_starts, allowed_assignments=allowed_assignments,
            alpha=config['alpha'], beta=config['beta'], rho=config['rho'],
            num_ants=config['num_ants'], iterations=config['iterations'],
            q=config['q'], time_limit=270.0
        )

        start_time = time.time()
        best_solution = aco.run()
        execution_time = time.time() - start_time

        fitness = calculate_fitness(best_solution.num_scheduled, best_solution.makespan,
                                 best_solution.total_cost, N)

        return {
            'config': config,
            'scheduled_tasks': best_solution.num_scheduled,
            'makespan': best_solution.makespan,
            'total_cost': best_solution.total_cost,
            'execution_time': execution_time,
            'fitness': fitness
        }
    except Exception as e:
        return {
            'config': config,
            'scheduled_tasks': 0,
            'makespan': float('inf'),
            'total_cost': float('inf'),
            'execution_time': 0,
            'fitness': float('inf'),
            'error': str(e)
        }

def test_single_pso_config(input_data, project_data, config):
    """Test a single PSO configuration"""
    try:
        start_time = time.time()
        best_solution = solve_pso(
            project_data,
            swarm_size=config['swarm_size'],
            max_iterations=config['max_iterations'],
            w=config['w'],
            c1=config['c1'],
            c2=config['c2'],
            time_limit=270.0
        )
        execution_time = time.time() - start_time

        if best_solution is None:
            return {
                'config': config,
                'scheduled_tasks': 0,
                'makespan': float('inf'),
                'total_cost': float('inf'),
                'execution_time': execution_time,
                'fitness': float('inf')
            }

        fitness = calculate_fitness(best_solution.scheduled_count, best_solution.makespan,
                                 best_solution.total_cost, project_data.num_tasks)

        return {
            'config': config,
            'scheduled_tasks': best_solution.scheduled_count,
            'makespan': best_solution.makespan,
            'total_cost': best_solution.total_cost,
            'execution_time': execution_time,
            'fitness': fitness
        }
    except Exception as e:
        return {
            'config': config,
            'scheduled_tasks': 0,
            'makespan': float('inf'),
            'total_cost': float('inf'),
            'execution_time': 0,
            'fitness': float('inf'),
            'error': str(e)
        }

def generate_aco_configs(num_configs=20):
    """Generate diverse ACO configurations"""
    configs = []

    # Define parameter ranges
    alphas = [0.5, 1.0, 1.5, 2.0]
    betas = [1.0, 2.0, 3.0, 4.0]
    rhos = [0.05, 0.1, 0.2, 0.3]
    num_ants = [10, 20, 30, 40, 50]
    iterations = [50, 100, 150]
    qs = [500, 1000, 1500, 2000]

    # Generate combinations
    for _ in range(num_configs):
        config = {
            'alpha': random.choice(alphas),
            'beta': random.choice(betas),
            'rho': random.choice(rhos),
            'num_ants': random.choice(num_ants),
            'iterations': random.choice(iterations),
            'q': random.choice(qs)
        }
        configs.append(config)

    return configs

def generate_pso_configs(num_configs=20):
    """Generate diverse PSO configurations"""
    configs = []

    # Define parameter ranges
    swarm_sizes = [20, 30, 50, 70, 100]
    max_iterations = [100, 150, 200]
    ws = [0.4, 0.6, 0.75, 0.8, 0.9]
    c1s = [0.5, 1.0, 1.5, 2.0]
    c2s = [0.5, 1.0, 1.5, 2.0]

    # Generate combinations
    for _ in range(num_configs):
        config = {
            'swarm_size': random.choice(swarm_sizes),
            'max_iterations': random.choice(max_iterations),
            'w': random.choice(ws),
            'c1': random.choice(c1s),
            'c2': random.choice(c2s)
        }
        configs.append(config)

    return configs

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_tuner.py <input_file> [algorithm] [configs]")
        print("Algorithm: 'aco', 'pso', or 'both' (default: both)")
        print("Configs: number of configurations to test (default: 15)")
        sys.exit(1)

    input_file = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "both"
    num_configs = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    print(f"Quick Hyperparameter Tuning")
    print(f"Input: {input_file}")
    print(f"Algorithm: {algorithm}")
    print(f"Configs per algorithm: {num_configs}")

    # Parse input data
    input_data = parse_input_file(input_file)
    project_data = convert_to_project_data(input_data)

    N = input_data[0]
    print(f"Problem size: {N} tasks")

    results = {}

    if algorithm in ["aco", "both"]:
        print("\n" + "="*50)
        print("TESTING ACO ALGORITHM")
        print("="*50)

        aco_configs = generate_aco_configs(num_configs)
        aco_results = []

        for i, config in enumerate(aco_configs, 1):
            print(f"\nACO Config {i}/{len(aco_configs)}: {config}")
            result = test_single_aco_config(input_data, config)
            aco_results.append(result)

            if 'error' in result:
                print(f"  ❌ FAILED: {result['error']}")
            else:
                print(f"  ✅ Scheduled: {result['scheduled_tasks']}/{N}")
                print(f"     Makespan: {result['makespan']:.1f}, Cost: {result['total_cost']:.1f}")
                print(f"     Fitness: {result['fitness']:.1f}, Time: {result['execution_time']:.1f}s")

        # Sort ACO results by fitness
        aco_results.sort(key=lambda x: x['fitness'])
        results['aco'] = aco_results

        print(f"\n{'='*50}")
        print("TOP 5 ACO CONFIGURATIONS")
        print("="*50)
        for i, result in enumerate(aco_results[:5], 1):
            print(f"\n{i}. Fitness: {result['fitness']:.1f}")
            print(f"   Config: {result['config']}")
            if 'error' not in result:
                print(f"   Results: Scheduled {result['scheduled_tasks']}/{N}, "
                      f"Makespan {result['makespan']:.1f}, Cost {result['total_cost']:.1f}")

    if algorithm in ["pso", "both"]:
        print("\n" + "="*50)
        print("TESTING PSO ALGORITHM")
        print("="*50)

        pso_configs = generate_pso_configs(num_configs)
        pso_results = []

        for i, config in enumerate(pso_configs, 1):
            print(f"\nPSO Config {i}/{len(pso_configs)}: {config}")
            result = test_single_pso_config(input_data, project_data, config)
            pso_results.append(result)

            if 'error' in result:
                print(f"  ❌ FAILED: {result['error']}")
            else:
                print(f"  ✅ Scheduled: {result['scheduled_tasks']}/{N}")
                print(f"     Makespan: {result['makespan']:.1f}, Cost: {result['total_cost']:.1f}")
                print(f"     Fitness: {result['fitness']:.1f}, Time: {result['execution_time']:.1f}s")

        # Sort PSO results by fitness
        pso_results.sort(key=lambda x: x['fitness'])
        results['pso'] = pso_results

        print(f"\n{'='*50}")
        print("TOP 5 PSO CONFIGURATIONS")
        print("="*50)
        for i, result in enumerate(pso_results[:5], 1):
            print(f"\n{i}. Fitness: {result['fitness']:.1f}")
            print(f"   Config: {result['config']}")
            if 'error' not in result:
                print(f"   Results: Scheduled {result['scheduled_tasks']}/{N}, "
                      f"Makespan {result['makespan']:.1f}, Cost {result['total_cost']:.1f}")

    # Comparison
    if 'aco' in results and 'pso' in results and results['aco'] and results['pso']:
        print(f"\n{'='*50}")
        print("ALGORITHM COMPARISON")
        print("="*50)

        best_aco = results['aco'][0]
        best_pso = results['pso'][0]

        print(f"\nBest ACO:")
        print(f"  Fitness: {best_aco['fitness']:.1f}")
        print(f"  Scheduled: {best_aco['scheduled_tasks']}/{N}")
        print(f"  Config: {best_aco['config']}")

        print(f"\nBest PSO:")
        print(f"  Fitness: {best_pso['fitness']:.1f}")
        print(f"  Scheduled: {best_pso['scheduled_tasks']}/{N}")
        print(f"  Config: {best_pso['config']}")

        if best_aco['fitness'] < best_pso['fitness']:
            print(f"\n🏆 ACO performs better on this dataset!")
        else:
            print(f"\n🏆 PSO performs better on this dataset!")

    # Save results
    timestamp = int(time.time())
    results_file = f"tuning_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()