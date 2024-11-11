import numpy as np
import heapq
import random
import plotly.graph_objects as go

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
    # Extract obstacle coordinates for plotting
    x_obs, y_obs, z_obs = np.where(grid == 1)

    # Initialize Plotly figure
    fig = go.Figure()

    # Plot obstacles with higher opacity for better visibility
    fig.add_trace(go.Scatter3d(
        x=x_obs, y=y_obs, z=z_obs,
        mode='markers',
        marker=dict(size=5, color='darkgray', opacity=0.4),
        name='Obstacles'
    ))

    # Plot start and end points with distinct colors and larger markers
    fig.add_trace(go.Scatter3d(
        x=[start[0]], y=[start[1]], z=[start[2]],
        mode='markers',
        marker=dict(size=12, color='limegreen', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[end[0]], y=[end[1]], z=[end[2]],
        mode='markers',
        marker=dict(size=12, color='royalblue', symbol='diamond'),
        name='End'
    ))

    # Plot path with bold color and increased line width
    if path:
        x_path, y_path, z_path = zip(*path)
        fig.add_trace(go.Scatter3d(
            x=x_path, y=y_path, z=z_path,
            mode='lines+markers',
            line=dict(color='red', width=6),
            marker=dict(size=6, color='orange', opacity=0.9),
            name='Path'
        ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="white"),
            yaxis=dict(title='Y', backgroundcolor="white"),
            zaxis=dict(title='Z', backgroundcolor="white")
        ),
        title=title,
        showlegend=True
    )

    fig.show()

# Parameters for different difficulty levels
difficulty_levels = {"Easy": 0.1, "Medium": 0.3, "Hard": 0.5}
start, end = (0, 0, 0), (19, 19, 9)

# Run for each difficulty level
for level, ratio in difficulty_levels.items():
    print(f"Testing {level} difficulty level...")
    grid = create_random_map(size=20, depth=10, obstacle_ratio=ratio, start=start, end=end)
    path = a_star_search(grid, start, end)

    if path:
        print(f"Path found with {len(path)} steps.")
        visualize_path_3d(grid, path, start, end, title=f"3D Path Planning ({level} Difficulty)")
    else:
        print(f"No path found for {level} difficulty level.")
