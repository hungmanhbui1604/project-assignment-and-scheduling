import sys
import random
import math
import time

SWARM_SIZE = 100
MAX_ITERATIONS = 200 # Increased slightly for better convergence
W = 0.4   # Inertia weight
C1 = 1.5 # Cognitive (personal best)
C2 = 2.0 # Social (global best)
TIME_LIMIT = 300.0  # Time limit in seconds

# ==========================================
# 1. DATA STRUCTURES & INPUT PARSING
# ==========================================

class ProjectData:
    def __init__(self):
        self.num_tasks = 0       # N
        self.durations = []      # d(i) (0-indexed)
        self.num_teams = 0       # M
        self.team_ready_times = [] # s(j) (0-indexed)

        # Graph / Precedence
        # adj[u] = [v, w...] means task u must finish before v and w start
        self.adj = []
        self.in_degree = []      # number of prerequisites for each task

        # Costs / Capabilities
        # valid_teams[task_id] = [(team_id, cost), (team_id, cost)...]
        self.valid_teams = {}

def parse_input():
    """
    Reads from stdin and returns structured data for scheduler.
    Returns:
        tuple: (N, M, durations, precedence, team_starts, allowed_assignments)
    """
    input_data = sys.stdin.read().split()
    if not input_data:
        return None

    iterator = iter(input_data)

    try:
        # --- Section 1: Task Constraints ---
        # Line 1: N (Tasks) and Q (Precedence Pairs)
        N = int(next(iterator))
        Q = int(next(iterator))

        # Line i + 1: Precedence constraints (i -> j)
        # We store predecessors: precedence[j] = [i, ...] means j needs i to finish
        precedence = {i: [] for i in range(1, N + 1)}
        for _ in range(Q):
            i = int(next(iterator))
            j = int(next(iterator))
            precedence[j].append(i)

        # Line Q + 2: Durations d(1)...d(N)
        durations = {}
        for i in range(1, N + 1):
            durations[i] = int(next(iterator))

        # --- Section 2: Team Constraints ---
        # Line Q + 3: M (Teams)
        M = int(next(iterator))

        # Line Q + 4: Team start times s(1)...s(M)
        team_starts = []
        for _ in range(M):
            team_starts.append(int(next(iterator)))

        # --- Section 3: Costs ---
        # Line Q + 5: K (number of allowed assignments)
        K_val = int(next(iterator))

        # Line Q + 5 + k: i (task), j (team), c (cost)
        allowed_assignments = {}
        for _ in range(K_val):
            task_i = int(next(iterator))
            team_j = int(next(iterator))
            cost = int(next(iterator))
            allowed_assignments[(task_i, team_j)] = cost

        return N, M, durations, precedence, team_starts, allowed_assignments

    except StopIteration:
        return None

def convert_to_project_data(input_data):
    """
    Converts data from parse_input() format to ProjectData format
    """
    if input_data is None:
        return ProjectData()

    N, M, durations, precedence, team_starts, allowed_assignments = input_data

    data = ProjectData()
    data.num_tasks = N
    data.num_teams = M
    data.team_ready_times = team_starts

    # Convert durations from dict to list (0-indexed)
    data.durations = [durations[i] for i in range(1, N + 1)]

    # Initialize graph structures
    data.adj = [[] for _ in range(N)]
    data.in_degree = [0] * N

    # Convert precedence from successors list to adjacency list and in_degree
    for j in range(1, N + 1):
        for pred in precedence[j]:
            # pred -> j (pred must finish before j starts)
            data.adj[pred - 1].append(j - 1)  # Convert to 0-based
            data.in_degree[j - 1] += 1

    # Convert allowed_assignments to valid_teams format
    for i in range(N):
        data.valid_teams[i] = []

    for (task_i, team_j), cost in allowed_assignments.items():
        # Convert to 0-based indices
        task_idx = task_i - 1
        team_idx = team_j - 1
        data.valid_teams[task_idx].append((team_idx, cost))

    return data

# ==========================================
# 2. CORE LOGIC: DECODING (PARTICLE -> SCHEDULE)
# ==========================================

class ScheduleResult:
    def __init__(self):
        self.assignments = [] # List of (task_id, team_id, start_time)
        self.makespan = 0
        self.total_cost = 0
        self.scheduled_count = 0

def decode_particle(particle_position, data):
    """
    Converts a continuous particle vector into a valid schedule using Serial SGS.
    """
    N = data.num_tasks

    # Split particle into two parts
    priorities = particle_position[:N]     # Determines order
    team_selectors = particle_position[N:] # Determines WHO does it

    # ---------------------------------------------------------
    # SIMULATION STATE
    # ---------------------------------------------------------

    # 1. Track current availability of each team
    #    initially = s(j) from input
    current_team_times = list(data.team_ready_times)

    # 2. Track dependencies
    #    We need a mutable copy of in-degrees to know when tasks become ready
    current_in_degree = list(data.in_degree)

    # 3. Track earliest start time allowed by predecessors
    #    valid_start_time_preds[u] = max(finish_time of all parents of u)
    valid_start_time_preds = [0] * N

    result = ScheduleResult()

    # 4. The "Ready List" initialization
    #    Tasks that have 0 prerequisites left.
    ready_tasks = [i for i, deg in enumerate(current_in_degree) if deg == 0]

    # ---------------------------------------------------------
    # SCHEDULING LOOP (Serial Schedule Generation Scheme)
    # ---------------------------------------------------------

    while ready_tasks:
        # STEP A: SELECT TASK
        # Pick the task from ready_tasks with the highest Priority Value in the particle
        selected_task = max(ready_tasks, key=lambda t: priorities[t])

        # Remove from ready list so we don't process it again
        ready_tasks.remove(selected_task)

        # STEP B: SELECT TEAM
        options = data.valid_teams[selected_task]

        if not options:
            # If a task has no capable teams, it cannot be scheduled.
            # We skip it, but this effectively blocks all its successors.
            continue

        # Use the "Spinner" logic to pick a team
        selector_value = team_selectors[selected_task]

        # Ensure selector is strictly < 1.0 for index calculation
        if selector_value >= 1.0: selector_value = 0.99999

        num_options = len(options)
        choice_index = int(math.floor(selector_value * num_options))

        assigned_team, task_cost = options[choice_index]

        # STEP C: CALCULATE START TIME
        # Constraint 1: Must wait for predecessors to finish
        start_lim_preds = valid_start_time_preds[selected_task]

        # Constraint 2: Must wait for the specific team to be free
        start_lim_team = current_team_times[assigned_team]

        # Actual start is the max of the two constraints
        actual_start_time = max(start_lim_preds, start_lim_team)
        actual_finish_time = actual_start_time + data.durations[selected_task]

        # STEP D: UPDATE STATE
        # 1. Update Team Availability
        current_team_times[assigned_team] = actual_finish_time

        # 2. Update Result Stats
        result.assignments.append({
            'task': selected_task,
            'team': assigned_team,
            'start': actual_start_time
        })
        result.total_cost += task_cost
        result.makespan = max(result.makespan, actual_finish_time)
        result.scheduled_count += 1

        # 3. Unlock Successors
        for neighbor in data.adj[selected_task]:
            current_in_degree[neighbor] -= 1

            # The neighbor cannot start until *this* task finishes
            # We update the neighbor's constraint to be at least this finish time
            valid_start_time_preds[neighbor] = max(valid_start_time_preds[neighbor], actual_finish_time)

            # If all parents are done, add to ready list
            if current_in_degree[neighbor] == 0:
                ready_tasks.append(neighbor)

    return result

# ==========================================
# 3. PARTICLE SWARM OPTIMIZATION
# ==========================================

def calculate_fitness(result, data):
    """
    Hierarchical Objective:
    1. Maximize Scheduled Tasks (Penalty if < N)
    2. Minimize Makespan
    3. Minimize Cost
    """

    # Weights for Hierarchical approach
    W1 = 10000000 # Huge penalty for unscheduled tasks (Priority 1)
    W2 = 1000     # Moderate penalty for time (Priority 2)
    W3 = 1        # Small penalty for cost (Priority 3)

    penalty_unscheduled = (data.num_tasks - result.scheduled_count) * W1
    score_time = result.makespan * W2
    score_cost = result.total_cost * W3

    total_fitness = penalty_unscheduled + score_time + score_cost
    return total_fitness

class Particle:
    def __init__(self, size):
        # Position: [Priorities (N) ... | TeamSelectors (N) ...]
        self.position = [random.random() for _ in range(size)]
        self.velocity = [0.0 for _ in range(size)]

        self.best_position = list(self.position)
        self.best_fitness = float('inf')
        self.best_result = None

def solve_pso(data, swarm_size, max_iterations, w, c1, c2, time_limit=TIME_LIMIT):
    # PSO Configuration
    # SWARM_SIZE = 50
    # MAX_ITERATIONS = 150 # Increased slightly for better convergence

    # Standard PSO coefficients
    # w = 0.729    # Inertia weight
    # c1 = 1.49445 # Cognitive (personal best)
    # c2 = 1.49445 # Social (global best)

    dim = 2 * data.num_tasks
    swarm = [Particle(dim) for _ in range(swarm_size)]

    global_best_position = None
    global_best_fitness = float('inf')
    global_best_result = None

    start_time = time.time()  # Record start time

    for iteration in range(max_iterations):
        # Check if time limit exceeded
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            # print(f"Time limit of {time_limit} seconds reached. Stopping early.", file=sys.stderr)
            break
        for particle in swarm:
            # 1. Decode & Evaluate
            result = decode_particle(particle.position, data)
            fitness = calculate_fitness(result, data)

            # 2. Update Personal Best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = list(particle.position)
                particle.best_result = result

            # 3. Update Global Best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = list(particle.position)
                global_best_result = result

        # 4. Move Particles
        for particle in swarm:
            for i in range(dim):
                r1 = random.random()
                r2 = random.random()

                # Velocity update
                vel_cognitive = c1 * r1 * (particle.best_position[i] - particle.position[i])
                vel_social = c2 * r2 * (global_best_position[i] - particle.position[i])

                particle.velocity[i] = (w * particle.velocity[i]) + vel_cognitive + vel_social

                # Position update
                particle.position[i] += particle.velocity[i]

                # Boundary clamping [0.0, 1.0]
                if particle.position[i] < 0.0:
                    particle.position[i] = 0.0
                    particle.velocity[i] *= -0.5 # Wall bounce
                elif particle.position[i] > 1.0:
                    particle.position[i] = 1.0
                    particle.velocity[i] *= -0.5 # Wall bounce

    return global_best_result

# ==========================================
# 4. MAIN OUTPUT
# ==========================================

if __name__ == "__main__":
    # 1. Read Data
    input_data = parse_input()
    project_data = convert_to_project_data(input_data)

    if project_data.num_tasks > 0:
        # 2. Run Optimization
        best_schedule = solve_pso(project_data, swarm_size=SWARM_SIZE, max_iterations=MAX_ITERATIONS, w=W, c1=C1, c2=C2, time_limit=TIME_LIMIT)

        # 3. Print Output
        if best_schedule:
            # Line 1: R (number of scheduled tasks)
            print(best_schedule.scheduled_count)

            # Sort by Task ID (i) for organized output
            sorted_assignments = sorted(best_schedule.assignments, key=lambda x: x['task'])

            for item in sorted_assignments:
                # Output format: i j u
                # +1 to convert internal 0-based index to 1-based output
                print(f"{item['task'] + 1} {item['team'] + 1} {item['start']}")