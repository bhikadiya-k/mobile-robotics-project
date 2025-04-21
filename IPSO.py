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

        for center, radius in obstacles:
            dist = point_to_segment_distance(center, points[i], points[i+1])
            safe_distance = radius + 0.5
            if dist < safe_distance:
                penalty = (safe_distance - dist) + penalty

    return total_length + penalty_weight * penalty

def compute_path_length(path, start, goal):
    waypoints = path.reshape(-1, 2)
    points = [start] + list(waypoints) + [goal]


    return sum(np.linalg.norm(points[i+1] - points[i]) for i in range(len(points)-1))

def compute_dynamic_inertia_weight(position, obstacles, w_max, w_min, d_threshold):
    d_min = float('inf')
    for center, radius in obstacles:
        d = np.linalg.norm(position - center) - radius
        d_min = min(d_min, d)
    d_min = max(d_min, 0.0)

    if d_min >= d_threshold:
        return w_max
    return w_min + (w_max - w_min) * (d_min / d_threshold)


def ipso_path_planning(num_particles, num_waypoints, bounds, max_iter,
                       w_max, w_min, c1, c2, obstacles, start, goal, d_threshold):
    dim = num_waypoints * 2
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))

    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    pbest_positions = positions.copy()

    pbest_fitness = np.array([fitness_function(p, obstacles, start, goal) for p in positions])
    gbest_index = np.argmin(pbest_fitness)

    gbest_position = pbest_positions[gbest_index].copy()

    gbest_fitness = pbest_fitness[gbest_index]

    convergence_curve = []

    for _ in range(max_iter):
        for i in range(num_particles):
            waypoints = positions[i].reshape(-1, 2)
            ws = [compute_dynamic_inertia_weight(wp, obstacles, w_max, w_min, d_threshold) for wp in waypoints]

            w_dyn = np.mean(ws)
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            velocities[i] = (w_dyn * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            
            positions[i] = np.clip(positions[i] + velocities[i], bounds[0], bounds[1])

            f = fitness_function(positions[i], obstacles, start, goal)

            if f < pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = f

                if f < gbest_fitness:
                    gbest_fitness = f
                    gbest_position = positions[i].copy()
        convergence_curve.append(gbest_fitness)

    return gbest_position, gbest_fitness, convergence_curve


def plot_ipso_path(best_path, obstacles, start, goal, bounds):
    waypoints = best_path.reshape(-1, 2)
    path_points = np.vstack((start, waypoints, goal))
    plt.figure(figsize=(8, 8))

    plt.plot(path_points[:, 0], path_points[:, 1], 'ro-',color='b', label='Planned Path')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')

    for center, radius in obstacles:
        plt.gca().add_patch(plt.Circle(center, radius, color='r', alpha=0.5))
        plt.gca().add_patch(plt.Circle(center, radius + 0.5, color='r', linestyle='--', fill=False))

        
    plt.xlim(bounds[0] - 1, bounds[1] + 1)
    plt.ylim(bounds[0] - 1, bounds[1] + 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('IPSO Path Planning')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_ipso_convergence(convergence_curve):
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_curve, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.title('IPSO Convergence Curve')
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
        (np.array([4.0, 3.0]), 1.0)]

    num_particles = 50
    num_waypoints = 5
     
    bounds = (0, 10)
    max_iter = 100
    w_max = 0.9; w_min = 0.4; c1 = 1.5; c2 = 1.5
    d_threshold = np.sqrt(10**2 + 10**2)

    t0 = time.time()
    best_path, best_fitness, curve = ipso_path_planning(
        num_particles, num_waypoints, bounds, max_iter,
        w_max, w_min, c1, c2, obstacles, start, goal, d_threshold
    )
    t1 = time.time()
    runtime_ms = (t1 - t0) * 1000
    path_length = compute_path_length(best_path, start, goal)
    avg_iter_time = runtime_ms / max_iter

    print(f"IPSO Best Fitness:       {best_fitness:.3f}")
    print(f"Path Length:             {path_length:.3f} m")
    print(f"Total Runtime:           {runtime_ms:.2f} ms")
    print(f"Average Time/Iteration:  {avg_iter_time:.2f} ms")

    plot_ipso_path(best_path, obstacles, start, goal, bounds)
    plot_ipso_convergence(curve)

