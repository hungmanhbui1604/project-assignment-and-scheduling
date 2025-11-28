import sys
import random
import time

ALPHA = 0.5
BETA = 3.0
RHO = 0.1
NUM_ANTS = 40
ITERATIONS = 100
Q = 1000.0
TIME_LIMIT = 300.0  # Time limit in seconds

# ==========================================
# 1. Input Handler
# ==========================================
def parse_input():
    """
    Reads from stdin and returns structured data for the scheduler.
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

# ==========================================
# 2. Ant Colony Classes
# ==========================================
class Ant:
    def __init__(self, num_tasks, num_teams):
        self.schedule = []  # List of (task_id, team_id, start_time)
        self.team_free_time = {}
        self.task_finish_time = {}

        # Objectives
        self.makespan = 0
        self.total_cost = 0
        self.num_scheduled = 0

class ACO_Scheduler:
    def __init__(self, N, M, durations, precedence, team_starts, allowed_assignments,
                 alpha, beta, rho, num_ants, iterations, q, time_limit=TIME_LIMIT):
        self.N = N
        self.M = M
        self.durations = durations
        self.precedence = precedence
        self.team_starts = team_starts
        self.allowed_assignments = allowed_assignments

        # Pre-calculate successors for graph traversal logic
        self.successors = {}
        self.indegree_base = {i: 0 for i in range(1, N + 1)}
        for task, preds in self.precedence.items():
            self.indegree_base[task] = len(preds)
            for p in preds:
                if p not in self.successors: self.successors[p] = []
                self.successors[p].append(task)

        # ACO Params
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_ants = num_ants
        self.iterations = iterations
        self.q = q
        self.time_limit = time_limit

        # Initialize Pheromones
        self.pheromones = {}
        for (t_id, tm_id) in self.allowed_assignments.keys():
            if t_id not in self.pheromones: self.pheromones[t_id] = {}
            self.pheromones[t_id][tm_id] = 1.0

    def calculate_heuristic(self, task, team, current_team_time, task_ready_time):
        duration = self.durations[task]
        cost = self.allowed_assignments[(task, team)]
        start_time = max(current_team_time, task_ready_time)
        finish_time = start_time + duration

        # Heuristic: Earlier finish time and Lower cost are better
        h_time = 1.0 / (finish_time + 1.0)
        h_cost = 1.0 / (cost + 1.0)
        return (h_time ** 1.5) * (h_cost ** 0.5)

    def run(self):
        best_global_ant = None
        start_time = time.time()  # Record start time

        for it in range(self.iterations):
            # Check if time limit exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.time_limit:
                # print(f"Time limit of {self.time_limit} seconds reached. Stopping early.", file=sys.stderr)
                break
            ants = []
            for _ in range(self.num_ants):
                ant = Ant(self.N, self.M)
                for m in range(1, self.M + 1):
                    ant.team_free_time[m] = self.team_starts[m-1]

                current_indegree = self.indegree_base.copy()
                available_tasks = [t for t, deg in current_indegree.items() if deg == 0]

                while available_tasks:
                    candidates = []
                    for task in available_tasks:
                        ready_time = 0
                        if task in self.precedence:
                            for pred in self.precedence[task]:
                                ready_time = max(ready_time, ant.task_finish_time.get(pred, 0))

                        if task in self.pheromones:
                            for team in self.pheromones[task]:
                                tau = self.pheromones[task][team]
                                eta = self.calculate_heuristic(task, team, ant.team_free_time[team], ready_time)
                                prob = (tau ** self.alpha) * (eta ** self.beta)
                                candidates.append({'task': task, 'team': team, 'prob': prob, 'ready': ready_time})

                    if not candidates: break

                    # Selection
                    total_prob = sum(c['prob'] for c in candidates)
                    if total_prob == 0: chosen = random.choice(candidates)
                    else:
                        r = random.uniform(0, total_prob)
                        cumsum = 0; chosen = candidates[-1]
                        for c in candidates:
                            cumsum += c['prob']
                            if r <= cumsum: chosen = c; break

                    # Execute Assignment
                    task, team, r_time = chosen['task'], chosen['team'], chosen['ready']
                    start = max(ant.team_free_time[team], r_time)
                    finish = start + self.durations[task]
                    cost = self.allowed_assignments[(task, team)]

                    ant.schedule.append((task, team, start))
                    ant.team_free_time[team] = finish
                    ant.task_finish_time[task] = finish
                    ant.total_cost += cost
                    ant.makespan = max(ant.makespan, finish)

                    available_tasks.remove(task)
                    if task in self.successors:
                        for succ in self.successors[task]:
                            current_indegree[succ] -= 1
                            if current_indegree[succ] == 0: available_tasks.append(succ)

                ant.num_scheduled = len(ant.schedule)
                ants.append(ant)

            # Find best ant in this iteration
            ants.sort(key=lambda x: (-x.num_scheduled, x.makespan, x.total_cost))
            iter_best = ants[0]

            # Update Global Best (Priority: Count > Time > Cost)
            if best_global_ant is None: best_global_ant = iter_best
            else:
                if iter_best.num_scheduled > best_global_ant.num_scheduled:
                    best_global_ant = iter_best
                elif iter_best.num_scheduled == best_global_ant.num_scheduled:
                    if iter_best.makespan < best_global_ant.makespan:
                        best_global_ant = iter_best
                    elif iter_best.makespan == best_global_ant.makespan:
                        if iter_best.total_cost < best_global_ant.total_cost:
                            best_global_ant = iter_best

            # Evaporation
            for t in self.pheromones:
                for tm in self.pheromones[t]:
                    self.pheromones[t][tm] *= (1.0 - self.rho)

            # Update Pheromones (Global Best)
            reward = self.q / (best_global_ant.makespan + 1.0)
            for (task, team, _) in best_global_ant.schedule:
                if task in self.pheromones and team in self.pheromones[task]:
                    self.pheromones[task][team] += reward

        return best_global_ant

# ==========================================
# 3. Main Execution
# ==========================================
def solve():
    # 1. Parse Data
    data = parse_input()
    if not data:
        return

    # Unpack the tuple
    N, M, durations, precedence, team_starts, allowed_assignments = data

    # 2. Run Algorithm
    aco = ACO_Scheduler(N, M, durations, precedence, team_starts, allowed_assignments,
                        alpha=ALPHA, beta=BETA, rho=RHO, num_ants=NUM_ANTS, iterations=ITERATIONS, q=Q, time_limit=TIME_LIMIT)
    best_solution = aco.run()

    # 3. Output Result
    print(len(best_solution.schedule))
    for (task, team, start_u) in best_solution.schedule:
        print(f"{task} {team} {start_u}")

if __name__ == "__main__":
    solve()