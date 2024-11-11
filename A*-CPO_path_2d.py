import numpy as np
import matplotlib.pyplot as plt
import heapq
import random


class Node:
    def __init__(self, x, y, g=0, h=0):
        self.x = x
        self.y = y
        self.g = g  # Path cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a.x - b.x) + abs(a.y - b.y)


def get_neighbors(node, grid):
    """Fetch valid neighbors that only allow horizontal or vertical moves."""
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = node.x + dx, node.y + dy
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
            neighbors.append((nx, ny))
    return neighbors


def a_star_search(grid, start, end):
    """Standard A* pathfinding."""
    open_set = []
    start_node = Node(*start, g=0, h=heuristic(Node(*start), Node(*end)))
    end_node = Node(*end)
    heapq.heappush(open_set, (start_node.f, start_node))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if (current.x, current.y) == (end_node.x, end_node.y):
            path = []
            while current:
                path.append((current.x, current.y))
                current = came_from.get((current.x, current.y))
            return path[::-1]

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[(current.x, current.y)] + 1
            if (neighbor[0], neighbor[1]) not in g_score or tentative_g_score < g_score[(neighbor[0], neighbor[1])]:
                came_from[(neighbor[0], neighbor[1])] = current
                g_score[(neighbor[0], neighbor[1])] = tentative_g_score
                f_score = tentative_g_score + heuristic(Node(*neighbor), end_node)
                heapq.heappush(open_set, (
                f_score, Node(neighbor[0], neighbor[1], tentative_g_score, heuristic(Node(*neighbor), end_node))))

    return None  # No path found


def cpo_optimized_path(path, grid, exploration_rate=0.1):
    """CPO-inspired light optimization that preserves only 90-degree turns."""
    optimized_path = path.copy()
    for i in range(1, len(optimized_path) - 1):
        prev = optimized_path[i - 1]
        current = optimized_path[i]
        next_node = optimized_path[i + 1]

        # Check if the current segment is a straight line (no turns).
        if (prev[0] == current[0] == next_node[0]) or (prev[1] == current[1] == next_node[1]):
            neighbors = get_neighbors(Node(*current), grid)

            # Only consider neighbors that maintain the straight path without altering right-angle turns
            valid_neighbors = [
                n for n in neighbors
                if (n[0] == current[0] == prev[0]) or (n[1] == current[1] == prev[1])
            ]

            # Randomly explore valid neighbors
            if valid_neighbors and random.random() < exploration_rate:
                selected = min(valid_neighbors, key=lambda node: heuristic(Node(*node), Node(*optimized_path[-1])))
                if heuristic(Node(*selected), Node(*optimized_path[-1])) < heuristic(Node(*current),
                                                                                     Node(*optimized_path[-1])):
                    optimized_path[i] = selected

    return optimized_path


def create_random_map(size=20, obstacle_ratio=0.3, start=(0, 0), end=(19, 19)):
    """Generate a random grid with obstacles."""
    grid = np.zeros((size, size))
    num_obstacles = int(size * size * obstacle_ratio)

    obstacles = set()
    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if (x, y) != start and (x, y) != end:
            obstacles.add((x, y))

    for (x, y) in obstacles:
        grid[x, y] = 1
    return grid


def visualize_path(grid, path, start, end, title="Path Planning"):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="binary", origin="lower")
    plt.plot(start[1], start[0], "go", markersize=10, label="Start")
    plt.plot(end[1], end[0], "bo", markersize=10, label="End")

    if path:
        x_coords, y_coords = zip(*path)
        plt.plot(y_coords, x_coords, color="red", linewidth=2, label="Path")
        plt.scatter(y_coords, x_coords, c=np.linspace(0, 1, len(path)), cmap="cool", s=50)

    plt.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    plt.show()


# Parameters and execution
start, end = (0, 0), (19, 19)
difficulty_levels = {"Easy": 0.2, "Medium": 0.3, "Hard": 0.4}

for level, ratio in difficulty_levels.items():
    grid = create_random_map(size=20, obstacle_ratio=ratio, start=start, end=end)
    path = a_star_search(grid, start, end)

    if path:
        print(f"{level} difficulty: Initial path length: {len(path)} steps")
        optimized_path = cpo_optimized_path(path, grid, exploration_rate=0.05)
        visualize_path(grid, optimized_path, start, end, title=f"Path Planning ({level} Difficulty)")
    else:
        print(f"{level} difficulty: No path found.")
