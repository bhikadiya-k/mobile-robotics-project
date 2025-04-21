import numpy as np
import matplotlib.pyplot as plt
import time


def point_to_segment_distance(p, a, b):
  
    ab = b - a
    if np.linalg.norm(ab) == 0:
        return np.linalg.norm(p - a)

    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    projection = a + t * ab
    return np.linalg.norm(p - projection)

def fitness_function(path, obstacles, start, goal, penalty_weight=100.0):
    waypoints = path.reshape(-1, 2)
    points = [start] + list(waypoints) + [goal]
    total_length = 0.0
    penalty = 0.0
    for i in range(len(points) - 1):
        seg_length = np.linalg.norm(points[i+1] - points[i])
        total_length = seg_length + total_length

        for (center, radius) in obstacles:
            dist = point_to_segment_distance(center, points[i], points[i+1])
            safe_distance = radius + 0.5
            if dist < safe_distance:

                penalty = (safe_distance - dist) + penalty

    return total_length + penalty_weight * penalty

def compute_path_length(path, start, goal):
    waypoints = path.reshape(-1, 2)
    points = [start] + list(waypoints) + [goal]
    return sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))


def pso_path_planning(num_particles, num_waypoints, bounds, max_iter, w, c1, c2, obstacles, start, goal):
    dim = num_waypoints * 2
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.random.uniform(-1, 1, (num_particles, dim))

    pbest_positions = positions.copy()
    pbest_fitness = np.array([fitness_function(pos, obstacles, start, goal) for pos in positions])

    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_index].copy()
    gbest_fitness = pbest_fitness[gbest_index]
    convergence_curve = []

    for iteration in range(max_iter):
        for i in range(num_particles):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            positions[i] = velocities[i] + positions[i]
            np.clip(positions[i], bounds[0], bounds[1], out=positions[i])

            current_fitness = fitness_function(positions[i], obstacles, start, goal)
            if current_fitness < pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = current_fitness
                if current_fitness < gbest_fitness:
                    gbest_position = positions[i].copy()
                    gbest_fitness = current_fitness
        convergence_curve.append(gbest_fitness)

    return gbest_position, gbest_fitness, convergence_curve

def plot_pso_path(best_path, obstacles, start, goal, bounds):
    waypoints = best_path.reshape(-1, 2)
    path_points = np.vstack((start, waypoints, goal))
    plt.figure(figsize=(8, 8))
    plt.plot(path_points[:, 0], path_points[:, 1], 'bo-', label='Planned Path')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    for (center, radius) in obstacles:
        plt.gca().add_patch(plt.Circle(center, radius, color='r', alpha=0.5))
        plt.gca().add_patch(plt.Circle(center, radius + 0.5, color='r', linestyle='--', fill=False))
        
    plt.ylim(bounds[0]-1, bounds[1]+1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PSO Path Planning')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pso_convergence(convergence_curve):
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_curve, marker='o', linestyle='-',color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('PSO Convergence Curve')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    start = np.array([0, 0])
    goal = np.array([10, 10])
    obstacles = [
        (np.array([2.5, 2.5]), 1.0),
        (np.array([3.0, 6.0]), 1.2),
        (np.array([5.5, 7.5]), 0.9),
        (np.array([7.5, 2.0]), 1.0),
        (np.array([7.0, 8.0]), 1.1),
        (np.array([4.0, 3.0]), 1.0)
    ]
    num_particles = 50
    num_waypoints = 5
    bounds = (0, 10)
    max_iter = 100
    w = 0.7
    c1 = 1.5
    c2 = 1.5

    t0 = time.time()
    best_path, best_fitness, convergence_curve = pso_path_planning(
        num_particles, num_waypoints, bounds, max_iter, w, c1, c2, obstacles, start, goal
    )
    t1 = time.time()
    runtime_ms = (t1 - t0) * 1000

    path_length = compute_path_length(best_path, start, goal)
    avg_iter_time = runtime_ms / max_iter

    print(f"PSO Best Fitness:       {best_fitness:.3f}")

    print(f"Path Length:            {path_length:.3f} m")

    print(f"Total Runtime:          {runtime_ms:.2f} ms")
    print(f"Average Time/Iteration: {avg_iter_time:.2f} ms")

    plot_pso_path(best_path, obstacles, start, goal, bounds)

    plot_pso_convergence(convergence_curve)

