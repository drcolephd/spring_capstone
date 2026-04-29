import random

class MazeGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[1 for _ in range(width)] for _ in range(height)]  # 1 for wall, 0 for path

    def generate_maze(self):
        """Generate a maze layout using recursive backtracking."""
        # Start carving from the second cell (1, 1) to keep outer walls intact
        self._carve_path(1, 1)
        
        # Make sure outer edges are walls
        for x in range(self.width):
            self.maze[0][x] = 1
            self.maze[self.height - 1][x] = 1
        for y in range(self.height):
            self.maze[y][0] = 1
            self.maze[y][self.width - 1] = 1

        return self.maze

    def _carve_path(self, x, y):
        """Recursively carve out a path in the maze."""
        # Define directions: (dx, dy) for right, down, left, up
        directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check bounds and if the cell at (nx, ny) is a wall
            if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and self.maze[ny][nx] == 1:
                # Carve the path between the current cell and next cell
                self.maze[ny][nx] = 0
                self.maze[y + dy // 2][x + dx // 2] = 0  # Carve the wall in between
                # Recursively carve from the new cell
                self._carve_path(nx, ny)

    def display_maze(self):
        """Print the maze layout to the console (for debugging purposes)."""
        for row in self.maze:
            print("".join(["#" if cell == 1 else " " for cell in row]))
