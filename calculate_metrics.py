import sys

def parse_input_file(filename):
    """
    Parse the input file to extract problem data needed for cost calculations.
    Returns: (N, M, durations, precedence, team_starts, allowed_assignments)
    """
    with open(filename, 'r') as f:
        content = f.read().split()
    
    if not content:
        return None
        
    iterator = iter(content)
    
    try:
        # Parse tasks and precedence
        N = int(next(iterator))
        Q = int(next(iterator))
        
        precedence = {i: [] for i in range(1, N + 1)}
        for _ in range(Q):
            i = int(next(iterator))
            j = int(next(iterator))
            precedence[j].append(i)
            
        # Parse durations
        durations = {}
        for i in range(1, N + 1):
            durations[i] = int(next(iterator))
            
        # Parse teams
        M = int(next(iterator))
        
        team_starts = []
        for _ in range(M):
            team_starts.append(int(next(iterator)))
            
        # Parse costs
        K_val = int(next(iterator))
        
        allowed_assignments = {}
        for _ in range(K_val):
            task_i = int(next(iterator))
            team_j = int(next(iterator))
            cost = int(next(iterator))
            allowed_assignments[(task_i, team_j)] = cost
            
        return N, M, durations, precedence, team_starts, allowed_assignments
        
    except StopIteration:
        return None

def parse_output_file(filename):
    """
    Parse the output file from either algorithm.
    Returns: (num_scheduled, assignments) where assignments is list of (task, team, start)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return 0, []
    
    num_scheduled = int(lines[0].strip())
    assignments = []
    
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 3:
            task = int(parts[0])
            team = int(parts[1])
            start = int(parts[2])
            assignments.append((task, team, start))
    
    return num_scheduled, assignments

def calculate_metrics(assignments, durations, allowed_assignments):
    """
    Calculate the three metrics from assignments.
    Returns: (num_tasks, makespan, total_cost)
    """
    if not assignments:
        return 0, 0, 0
    
    # Number of tasks is just the count of assignments
    num_tasks = len(assignments)
    
    # Calculate makespan (maximum finish time)
    makespan = 0
    total_cost = 0
    
    for task, team, start in assignments:
        # Calculate finish time for this task
        finish_time = start + durations[task]
        makespan = max(makespan, finish_time)
        
        # Add cost for this assignment
        if (task, team) in allowed_assignments:
            total_cost += allowed_assignments[(task, team)]
        else:
            print(f"Warning: Cost not found for task {task}, team {team}")
    
    return num_tasks, makespan, total_cost

def print_metrics(num_tasks, makespan, total_cost):
    """
    Print the three metrics in a formatted way.
    """
    print("\n" + "="*50)
    print("SCHEDULING METRICS")
    print("="*50)
    print(f"Number of tasks scheduled: {num_tasks}")
    print(f"Completion time (makespan): {makespan}")
    print(f"Total cost: {total_cost}")
    print("="*50)

def main():
    if len(sys.argv) != 3:
        print("Usage: python calculate_metrics.py <input_file> <output_file>")
        print("Example: python calculate_metrics.py input.txt output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if files exist
    try:
        with open(input_file, 'r'):
            pass
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    try:
        with open(output_file, 'r'):
            pass
    except FileNotFoundError:
        print(f"Error: Output file '{output_file}' not found.")
        sys.exit(1)
    
    # Parse input file to get problem data
    print(f"Parsing input file: {input_file}")
    input_data = parse_input_file(input_file)
    if input_data is None:
        print(f"Error: Could not parse input file '{input_file}'.")
        sys.exit(1)
    
    N, M, durations, precedence, team_starts, allowed_assignments = input_data
    print(f"Input file contains: {N} tasks, {M} teams, {len(allowed_assignments)} possible assignments")
    
    # Parse output file to get assignments
    print(f"Parsing output file: {output_file}")
    num_scheduled, assignments = parse_output_file(output_file)
    print(f"Output file contains: {num_scheduled} scheduled tasks")
    
    # Calculate metrics
    print("Calculating metrics...")
    num_tasks, makespan, total_cost = calculate_metrics(assignments, durations, allowed_assignments)
    
    # Print results
    print_metrics(num_tasks, makespan, total_cost)

if __name__ == "__main__":
    main()