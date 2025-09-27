import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 3D Environment Setup ---
class Environment:
    def __init__(self, size, start, end, num_obstacles, obstacle_size_range):
        self.size = size
        self.start = start
        self.end = end
        self.num_obstacles = num_obstacles
        self.obstacle_size_range = obstacle_size_range
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            size = np.random.uniform(self.obstacle_size_range[0], self.obstacle_size_range[1], 3)
            position = np.random.uniform(0, self.size - size[0], 3)
            obstacles.append(np.array([position, position + size]))
        return obstacles

    def is_collision(self, point):
        for obs in self.obstacles:
            if np.all(point >= obs[0]) and np.all(point <= obs[1]):
                return True
        return False

    def plot(self, ax):
        # Plot obstacles
        for obs in self.obstacles:
            X, Y, Z = self._get_cube_verts(obs)
            ax.plot_surface(X, Y, Z, color='r', alpha=0.5, rstride=1, cstride=1)
        # Plot start and end points
        ax.scatter(*self.start, color='g', s=100, label='Start')
        ax.scatter(*self.end, color='b', s=100, label='End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_zlim(0, self.size)
        ax.legend()

    def _get_cube_verts(self, obs):
        low = obs[0]
        high = obs[1]
        x = np.array([[low[0], high[0], high[0], low[0], low[0]],
                      [low[0], high[0], high[0], low[0], low[0]],
                      [low[0], high[0], high[0], low[0], low[0]],
                      [low[0], high[0], high[0], low[0], low[0]]])
        y = np.array([[low[1], low[1], high[1], high[1], low[1]],
                      [low[1], low[1], high[1], high[1], low[1]],
                      [low[1], low[1], low[1], low[1], low[1]],
                      [high[1], high[1], high[1], high[1], high[1]]])
        z = np.array([[low[2], low[2], low[2], low[2], low[2]],
                      [high[2], high[2], high[2], high[2], high[2]],
                      [low[2], low[2], high[2], high[2], low[2]],
                      [low[2], low[2], high[2], high[2], low[2]]])
        # This is a bit of a hack to draw the cube sides.
        # A more robust solution would draw the 6 faces separately.
        X = np.array([x[0], x[1], [x[0,0],x[0,0],x[0,3],x[0,3],x[0,0]], [x[0,1],x[0,1],x[0,2],x[0,2],x[0,1]]])
        Y = np.array([y[0], y[1], [y[0,0],y[0,0],y[0,3],y[0,3],y[0,0]], [y[0,1],y[0,1],y[0,2],y[0,2],y[0,1]]])
        Z = np.array([[z[0,0],z[0,0],z[0,0],z[0,0],z[0,0]],
                      [z[1,0],z[1,0],z[1,0],z[1,0],z[1,0]],
                      [z[0,0],z[1,0],z[1,0],z[0,0],z[0,0]],
                      [z[0,0],z[1,0],z[1,0],z[0,0],z[0,0]]])
        return X, Y, Z

# --- ACO Algorithm ---
class ACO:
    def __init__(self, env, num_ants, alpha, beta, evaporation_rate, q):
        self.env = env
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.nodes = self._discretize_env()
        self.num_nodes = len(self.nodes)
        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        self.start_node_idx = 0
        self.end_node_idx = self.num_nodes - 1
        self.global_best_path_indices = None
        self.global_best_length = np.inf


    def _discretize_env(self, resolution=5, num_points=100):
        nodes = [self.env.start]
        while len(nodes) < num_points -1:
            point = np.random.uniform(0, self.env.size, 3)
            if not self.env.is_collision(point):
                # Check if the point is too close to existing nodes
                if all(np.linalg.norm(point - existing_node) > 0.5 for existing_node in nodes):
                    nodes.append(point)
        nodes.append(self.env.end)
        return np.array(nodes)

    def run_iterations(self, num_iterations): # Renamed from run
        # This method now runs for a given number of iterations and updates the global best path
        for iteration in range(num_iterations):
            all_paths = []
            for _ in range(self.num_ants):
                path_indices = self._construct_solution()
                if path_indices and path_indices[-1] == self.end_node_idx:
                    path_length = self._path_length_from_indices(path_indices)
                    all_paths.append((path_indices, path_length))

            if not all_paths:
                print(f"ACO Iteration: No valid paths found.")
                continue

            self._update_pheromone(all_paths)

            iteration_best_path, iteration_best_length = min(all_paths, key=lambda x: x[1])
            if iteration_best_length < self.global_best_length:
                self.global_best_path_indices = iteration_best_path
                self.global_best_length = iteration_best_length
            
            print(f"ACO Iteration: Best Length = {self.global_best_length:.2f}")

        if self.global_best_path_indices:
            best_path_coords = [self.nodes[i] for i in self.global_best_path_indices]
            return best_path_coords, self.global_best_length
        else:
            return None, np.inf

    def _construct_solution(self):
        path = [self.start_node_idx]
        current_node = self.start_node_idx
        visited = {current_node}

        while current_node != self.end_node_idx and len(path) < self.num_nodes:
            probabilities = self._calculate_probabilities(current_node, visited)
            if np.sum(probabilities) == 0:
                return None # Ant is stuck
            
            next_node = np.random.choice(self.num_nodes, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node

        if path[-1] != self.end_node_idx:
            return None # Path did not reach the end

        return path

    def _calculate_probabilities(self, current_node, visited):
        pheromone = self.pheromone[current_node].copy()
        heuristic = 1.0 / (np.linalg.norm(self.nodes - self.nodes[current_node], axis=1) + 1e-10)

        # Mask visited nodes
        for i in visited:
            pheromone[i] = 0

        # Check for collision-free paths
        for i in range(self.num_nodes):
            if i not in visited:
                if self.env.is_collision(self.nodes[i]): # Check node collision
                    pheromone[i] = 0
                # More advanced: check edge collision
                # mid_point = (self.nodes[current_node] + self.nodes[i]) / 2
                # if self.env.is_collision(mid_point):
                #     pheromone[i] = 0

        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities /= sum_probs
        else:
            return np.zeros_like(probabilities)
            
        return probabilities

    def _update_pheromone(self, all_paths):
        self.pheromone *= (1 - self.evaporation_rate)
        for path, length in all_paths:
            pheromone_deposit = self.q / length
            for i in range(len(path) - 1):
                start_idx, end_idx = path[i], path[i+1]
                self.pheromone[start_idx, end_idx] += pheromone_deposit
                self.pheromone[end_idx, start_idx] += pheromone_deposit # Symmetric pheromone

    def reinforce_path(self, path_coords, length):
        """Reinforces the pheromone trail for a given path (e.g., from GA)."""
        # This is a new method to allow GA to influence ACO
        
        # Need to convert path_coords back to node indices. This is tricky.
        # For now, let's find the closest nodes in our discretized set.
        path_indices = []
        for point in path_coords:
            distances = np.linalg.norm(self.nodes - point, axis=1)
            closest_node_idx = np.argmin(distances)
            if not path_indices or closest_node_idx != path_indices[-1]:
                 path_indices.append(closest_node_idx)

        # Update global best if this path is better
        if length < self.global_best_length:
            self.global_best_path_indices = path_indices
            self.global_best_length = length

        # Deposit extra pheromone on this high-quality path
        pheromone_deposit = self.q * 2 / length # Extra reinforcement
        for i in range(len(path_indices) - 1):
            start_idx, end_idx = path_indices[i], path_indices[i+1]
            self.pheromone[start_idx, end_idx] += pheromone_deposit
            self.pheromone[end_idx, start_idx] += pheromone_deposit

    def _path_length_from_indices(self, path_indices):
        length = 0
        for i in range(len(path_indices) - 1):
            p1 = self.nodes[path_indices[i]]
            p2 = self.nodes[path_indices[i+1]]
            length += np.linalg.norm(p2 - p1)
        return length

    def _path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length

# --- GA Algorithm ---
class GA:
    def __init__(self, env, population_size, num_generations, mutation_rate, tournament_size):
        self.env = env
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.population = [] # Initialized as empty

    def _initialize_population(self, base_path): # Takes a base_path
        population = [base_path]
        for _ in range(self.population_size - 1):
            # Create variations of the initial path
            new_path = base_path.copy() # FIX: Use base_path instead of self.initial_path
            # Simple mutation: randomly perturb some points
            for i in range(1, len(new_path) - 1):
                if np.random.rand() < 0.3: # 30% chance to perturb
                    perturbation = np.random.uniform(-1, 1, 3)
                    new_point = new_path[i] + perturbation
                    if not self.env.is_collision(new_point):
                        new_path[i] = new_point
            population.append(new_path)
        return population

    def optimize_path(self, path_to_optimize): # Renamed from run
        self.population = self._initialize_population(path_to_optimize)

        for generation in range(self.num_generations):
            # Calculate fitness for each individual
            fitness_scores = [1 / self._path_length(p) for p in self.population]

            new_population = []
            # Elitism: keep the best individual
            best_idx = np.argmax(fitness_scores)
            new_population.append(self.population[best_idx])

            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)

                # Crossover
                child = self._crossover(parent1, parent2)

                # Mutation
                child = self._mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
            best_len = self._path_length(self.population[np.argmax(fitness_scores)])
            print(f"GA Generation {generation}: Best Path Length = {best_len:.2f}")

        best_path = self.population[np.argmax([1 / self._path_length(p) for p in self.population])]
        return best_path, self._path_length(best_path)

    def _tournament_selection(self, fitness_scores):
        selection_ix = np.random.randint(len(self.population), size=self.tournament_size)
        best_ix = max(selection_ix, key=lambda i: fitness_scores[i])
        return self.population[best_ix]

    def _crossover(self, parent1, parent2):
        # Single point crossover
        min_len = min(len(parent1), len(parent2))
        if min_len < 2:
            return parent1.copy()
        
        crossover_point = np.random.randint(1, min_len -1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def _mutate(self, path):
        mutated_path = path.copy()
        for i in range(1, len(mutated_path) - 1):
            if np.random.rand() < self.mutation_rate:
                perturbation = np.random.uniform(-0.5, 0.5, 3)
                new_point = mutated_path[i] + perturbation
                # Ensure the new point is within bounds and not in collision
                if not self.env.is_collision(new_point) and np.all(new_point >= 0) and np.all(new_point < self.env.size):
                    mutated_path[i] = new_point
        return mutated_path

    def _path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length

# --- Main Execution and Visualization ---
def main():
    # Environment Parameters
    ENV_SIZE = 10
    START_POINT = np.array([0.5, 0.5, 0.5])
    END_POINT = np.array([9.5, 9.5, 9.5])
    NUM_OBSTACLES = 20
    OBSTACLE_SIZE_RANGE = (0.5, 2.0)

    # ACO Parameters
    ACO_ANTS = 20
    ACO_ALPHA = 1.0
    ACO_BETA = 5.0
    ACO_EVAPORATION = 0.5
    ACO_Q = 100

    # GA Parameters
    GA_POP_SIZE = 50
    GA_MUTATION_RATE = 0.1
    GA_TOURNAMENT_SIZE = 5

    # Hybrid Algorithm Parameters
    HYBRID_ITERATIONS = 10
    ACO_ITERATIONS_PER_HYBRID = 5
    GA_GENERATIONS_PER_HYBRID = 5 # Reduced for faster cycles

    # 1. Setup Environment
    env = Environment(ENV_SIZE, START_POINT, END_POINT, NUM_OBSTACLES, OBSTACLE_SIZE_RANGE)

    # 2. Initialize Algorithms
    aco = ACO(env, ACO_ANTS, ACO_ALPHA, ACO_BETA, ACO_EVAPORATION, ACO_Q)
    ga = GA(env, GA_POP_SIZE, GA_GENERATIONS_PER_HYBRID, GA_MUTATION_RATE, GA_TOURNAMENT_SIZE)

    # 3. Run Hybrid ACO-GA Algorithm
    print("--- Running Hybrid ACO-GA Algorithm ---")
    
    # Initial ACO run to get a starting path
    print("\n--- Initial ACO Run ---")
    best_path, best_length = aco.run_iterations(num_iterations=10)
    if best_path is None:
        print("Initial ACO failed to find a path. Exiting.")
        return
    
    initial_aco_path = best_path
    initial_aco_length = best_length
    print(f"Initial ACO Path Length: {best_length:.2f}")

    for i in range(HYBRID_ITERATIONS):
        print(f"\n--- Hybrid Iteration {i+1}/{HYBRID_ITERATIONS} ---")

        # Run GA to optimize the current best path
        print("--- Running GA ---")
        optimized_path, optimized_length = ga.optimize_path(best_path)
        
        if optimized_length < best_length:
            print(f"GA found a better path: {optimized_length:.2f}")
            best_path = optimized_path
            best_length = optimized_length
        else:
            print("GA did not improve the path.")

        # Reinforce ACO with the best path found so far
        print("--- Reinforcing ACO Pheromones ---")
        aco.reinforce_path(best_path, best_length)

        # Run ACO for a few more iterations
        print("--- Running ACO ---")
        aco_path, aco_length = aco.run_iterations(num_iterations=ACO_ITERATIONS_PER_HYBRID)
        
        if aco_path and aco_length < best_length:
            print(f"ACO found a better path: {aco_length:.2f}")
            best_path = aco_path
            best_length = aco_length
        else:
            print("ACO did not improve the path in this cycle.")

    print(f"\nHybrid algorithm finished. Final best path length: {best_length:.2f}")

    # 4. Visualization
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    env.plot(ax)

    # Plot Initial ACO path
    initial_aco_path_np = np.array(initial_aco_path)
    ax.plot(initial_aco_path_np[:, 0], initial_aco_path_np[:, 1], initial_aco_path_np[:, 2], 'y--', label=f'Initial ACO Path ({initial_aco_length:.2f})')

    # Plot Final Hybrid path
    final_path_np = np.array(best_path)
    ax.plot(final_path_np[:, 0], final_path_np[:, 1], final_path_np[:, 2], 'c-', linewidth=2, label=f'Hybrid ACO-GA Path ({best_length:.2f})')

    ax.set_title('Hybrid ACO-GA 3D Path Planning')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()