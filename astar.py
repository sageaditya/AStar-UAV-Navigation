import numpy as np
from typing import List, Tuple, Optional, Union
import heapq

class Node:
    def __init__(self, position: np.ndarray):
        self.position = position
        self.g_cost = float('inf')
        self.h_cost = 0.0
        self.f_cost = float('inf')
        self.parent = None

    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarBasicPlanner:
    def __init__(self, 
                 start: np.ndarray,
                 goal: np.ndarray,
                 bounds: np.ndarray,
                 obstacles: List[Union[np.ndarray, Tuple, dict]],
                 max_iterations: int = 15000):
        
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.nodes = []
        self.max_iterations = max_iterations

    def check_collision(self, point: np.ndarray) -> bool:
        for obs in self.obstacles:
            if isinstance(obs, np.ndarray):
                x, y, z = point
                ox_min, oy_min, oz_min, ox_max, oy_max, oz_max = obs
                if (ox_min <= x <= ox_max and
                    oy_min <= y <= oy_max and
                    oz_min <= z <= oz_max):
                    return True
            else:
                if isinstance(obs, dict):
                    center = obs['position']
                    radius = obs['radius']
                else:
                    center, radius = obs
                if np.linalg.norm(point - np.array(center)) <= radius:
                    return True
        return False

    def get_neighbors(self, point: np.ndarray) -> List[np.ndarray]:
        step_size = 1.0
        neighbors = []
        
        # Basic 6-directional movement
        directions = [
            [step_size, 0, 0], [-step_size, 0, 0],
            [0, step_size, 0], [0, -step_size, 0],
            [0, 0, step_size], [0, 0, -step_size]
        ]
        
        for dx, dy, dz in directions:
            new_point = point + np.array([dx, dy, dz])
            
            # Check bounds
            if not all(self.bounds[i][0] <= new_point[i] <= self.bounds[i][1] for i in range(3)):
                continue
                
            # Check collision
            if self.check_collision(new_point):
                continue
                
            neighbors.append(new_point)
            
        return neighbors

    def heuristic(self, point: np.ndarray) -> float:
        return np.linalg.norm(point - self.goal)

    def smooth_path(self, path: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Smooth the path using a moving average filter while ensuring obstacle avoidance.
        
        Args:
            path: The raw path from A* algorithm
            window_size: Size of the moving average window (must be odd)
            
        Returns:
            Smoothed path that still avoids obstacles
        """
        if len(path) <= 2:
            return path
            
        # Ensure window_size is odd
        window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
        half_window = window_size // 2
        
        smoothed_path = [path[0]]  # Keep start point unchanged
        
        for i in range(1, len(path) - 1):
            # Get points in the window
            start_idx = max(0, i - half_window)
            end_idx = min(len(path), i + half_window + 1)
            window_points = path[start_idx:end_idx]
            
            # Calculate average position
            avg_point = np.mean(window_points, axis=0)
            
            # Check if the smoothed point is collision-free
            if not self.check_collision(avg_point):
                smoothed_path.append(avg_point)
            else:
                # If collision, keep original point
                smoothed_path.append(path[i])
        
        smoothed_path.append(path[-1])  # Keep end point unchanged
        return np.array(smoothed_path)

    def plan_path(self) -> Optional[np.ndarray]:
        start_node = Node(self.start)
        start_node.g_cost = 0
        start_node.h_cost = self.heuristic(self.start)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        node_dict = {tuple(self.start): start_node}
        
        iteration = 0
        best_path = None
        best_path_length = float('inf')
        
        print(f"Starting A* with max iterations: {self.max_iterations}")
        
        while open_set and iteration < self.max_iterations:
            iteration += 1
            if iteration % 500 == 0:  # Print progress every 500 iterations
                print(f"Current iteration: {iteration}")
                
            current = heapq.heappop(open_set)
            current_pos = tuple(current.position)
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            self.nodes.append(current)
            
            # Check if goal is reached
            if np.linalg.norm(current.position - self.goal) < 1.0:
                path = []
                temp_current = current
                while temp_current is not None:
                    path.append(temp_current.position)
                    temp_current = temp_current.parent
                current_path = np.array(path[::-1])
                
                # Calculate path length
                path_length = sum(np.linalg.norm(current_path[i] - current_path[i-1]) 
                                for i in range(1, len(current_path)))
                
                if path_length < best_path_length:
                    best_path = current_path
                    best_path_length = path_length
                    print(f"Found better path at iteration {iteration} with length: {path_length:.2f}")
            
            # Explore neighbors
            for neighbor_pos in self.get_neighbors(current.position):
                neighbor_tuple = tuple(neighbor_pos)
                
                if neighbor_tuple in closed_set:
                    continue
                    
                new_g_cost = current.g_cost + np.linalg.norm(neighbor_pos - current.position)
                
                if neighbor_tuple not in node_dict:
                    neighbor = Node(neighbor_pos)
                    node_dict[neighbor_tuple] = neighbor
                else:
                    neighbor = node_dict[neighbor_tuple]
                    
                if new_g_cost >= neighbor.g_cost:
                    continue
                    
                neighbor.parent = current
                neighbor.g_cost = new_g_cost
                neighbor.h_cost = self.heuristic(neighbor_pos)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                
                heapq.heappush(open_set, neighbor)
        
        print(f"Completed all {self.max_iterations} iterations")
        if best_path is not None:
            # Apply path smoothing
            smoothed_path = self.smooth_path(best_path)
            smoothed_length = sum(np.linalg.norm(smoothed_path[i] - smoothed_path[i-1]) 
                                for i in range(1, len(smoothed_path)))
            print(f"Original path length: {best_path_length:.2f}")
            print(f"Smoothed path length: {smoothed_length:.2f}")
            return smoothed_path
        return None