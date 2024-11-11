import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import heapq
import random

class Node:
    def __init__(self, x, y, z, g=0, h=0):
        self.x = x
        self.y = y
        self.z = z
        self.g = g  # Path cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    """Manhattan distance heuristic in 3D."""
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z)

def get_neighbors(node, grid):
    """Fetch valid neighbors that only allow moves along one axis at a time in 3D."""
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    neighbors = []
    for dx, dy, dz in directions:
        nx, ny, nz = node.x + dx, node.y + dy, node.z + dz
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2] and grid[nx, ny, nz] == 0:
            neighbors.append((nx, ny, nz))
    return neighbors

def a_star_search(grid, start, end):
    """A* pathfinding adapted for 3D."""
    open_set = []
    start_node = Node(*start, g=0, h=heuristic(Node(*start), Node(*end)))
    end_node = Node(*end)
    heapq.heappush(open_set, (start_node.f, start_node))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if (current.x, current.y, current.z) == (end_node.x, end_node.y, end_node.z):
            path = []
            while current:
                path.append((current.x, current.y, current.z))
                current = came_from.get((current.x, current.y, current.z))
            return path[::-1]

        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[(current.x, current.y, current.z)] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(Node(*neighbor), end_node)
                heapq.heappush(open_set, (f_score, Node(*neighbor, tentative_g_score, heuristic(Node(*neighbor), end_node))))

    return None  # No path found

def create_random_map(size=20, depth=10, obstacle_ratio=0.3, start=(0, 0, 0), end=(19, 19, 9)):
    """Generate a random 3D grid with obstacles."""
    grid = np.zeros((size, size, depth))
    num_obstacles = int(size * size * depth * obstacle_ratio)

    obstacles = set()
    while len(obstacles) < num_obstacles:
        x, y, z = random.randint(0, size - 1), random.randint(0, size - 1), random.randint(0, depth - 1)
        if (x, y, z) != start and (x, y, z) != end:
            obstacles.add((x, y, z))

    for (x, y, z) in obstacles:
        grid[x, y, z] = 1
    return grid

def visualize_path_3d(grid, path, start, end, title="3D Path Planning"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Set camera angle for better visualization
    ax.view_init(elev=30, azim=45)

    # Plot obstacles with light transparency to avoid clutter
    ax.voxels(grid == 1, facecolors='grey', edgecolor='none', alpha=0.2)

    # Highlight start and end points
    ax.scatter(*start, color='limegreen', s=100, label="Start", marker='o', edgecolor='black')
    ax.scatter(*end, color='royalblue', s=100, label="End", marker='o', edgecolor='black')

    # Plot path with gradient effect
    if path:
        x_coords, y_coords, z_coords = zip(*path)
        for i in range(len(x_coords) - 1):
            ax.plot([x_coords[i], x_coords[i + 1]],
                    [y_coords[i], y_coords[i + 1]],
                    [z_coords[i], z_coords[i + 1]],
                    color=plt.cm.plasma(i / len(x_coords)), linewidth=2.5)
        ax.scatter(x_coords, y_coords, z_coords, c=np.linspace(0, 1, len(path)), cmap="plasma", s=20, depthshade=True)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc='upper left')
    plt.show()

# Parameters for each difficulty level
difficulty_levels = {"Easy": 0.1, "Medium": 0.15, "Hard": 0.2}
start, end = (0, 0, 0), (19, 19, 9)

# Generate maps and visualize paths for each difficulty level
for level, ratio in difficulty_levels.items():
    print(f"\nTesting {level} difficulty level with obstacle ratio {ratio}")
    grid = create_random_map(size=20, depth=10, obstacle_ratio=ratio, start=start, end=end)
    path = a_star_search(grid, start, end)

    if path:
        print(f"Path found for {level} difficulty with {len(path)} steps.")
        visualize_path_3d(grid, path, start, end, title=f"3D Path Planning ({level} Difficulty)")
    else:
        print(f"No path found for {level} difficulty level.")
