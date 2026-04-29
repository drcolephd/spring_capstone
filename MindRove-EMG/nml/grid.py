from PyQt5 import QtCore

class Grid:
    def __init__(self, num_rows, num_columns, cell_size=10, border_thickness=1):
        """Initialize grid properties for positioning calculations only.
        
        Parameters:
        - num_rows: The number of rows in the grid.
        - num_columns: The number of columns in the grid.
        - cell_size: The size of each cell in pixels.
        - border_thickness: Thickness of the border.
        """
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.cell_size = cell_size
        self.border_thickness = border_thickness
        
        # Sets to store walls
        self.horizontal_walls = set()  # {(row, col), (row, col+1)}
        self.vertical_walls = set()    # {(row, col), (row+1, col)}

    def add_wall(self, row0, col0, row1, col1):
        """Add a wall between two cells (row0, col0) and (row1, col1).
        
        Walls can only be added between adjacent cells and aligned horizontally or vertically.
        """
        if row0 == row1 and abs(col0 - col1) == 1:
            # Horizontal wall between (row, col) and (row, col+1)
            self.horizontal_walls.add((min(row0, row1), min(col0, col1)))
        elif col0 == col1 and abs(row0 - row1) == 1:
            # Vertical wall between (row, col) and (row+1, col)
            self.vertical_walls.add((min(row0, row1), min(col0, col1)))
        else:
            raise ValueError("Walls must be between adjacent cells and aligned either horizontally or vertically.")

    def is_wall_between(self, start_row, start_col, end_row, end_col):
        """Check if there is a wall between two cells."""
        if start_row == end_row:
            # Horizontal movement
            return (min(start_row, end_row), min(start_col, end_col)) in self.horizontal_walls
        elif start_col == end_col:
            # Vertical movement
            return (min(start_row, end_row), min(start_col, end_col)) in self.vertical_walls
        return False

    def cell(self, row, column, w=20, h=20):
        """Calculate the top-left coordinate for a cell, accounting for border thickness."""
        cell_x = column * (self.cell_size + self.border_thickness) + (self.cell_size - w)/2
        cell_y = row * (self.cell_size + self.border_thickness) + (self.cell_size - h) / 2
        return QtCore.QPointF(cell_x, cell_y)

    def move(self, start_row, start_col, new_row, new_col):
        """Move within the grid if no walls block the path, respecting grid boundaries."""
        # Clamp new position to grid bounds
        row = min(max(new_row, 0), self.num_rows - 1)
        col = min(max(new_col, 0), self.num_columns - 1)
        
        # Check if there's a wall blocking the move
        if self.is_wall_between(start_row, start_col, row, col):
            # If movement is blocked by a wall, return original position
            return (self.cell(start_row, start_col), start_row, start_col)
        
        # If no wall is blocking, return the new position
        return (self.cell(row, col), row, col)

    def border(self, row0, col0, row1, col1):
        """Calculate the QRectF for a border line between two cells."""
        if col0 == col1:  # Vertical border
            border_x = col0 * (self.cell_size + self.border_thickness)
            border_y = min(row0, row1) * (self.cell_size + self.border_thickness)
            height = abs(row1 - row0) * (self.cell_size + self.border_thickness) + self.border_thickness
            return QtCore.QRectF(border_x, border_y, self.border_thickness, height)
        
        elif row0 == row1:  # Horizontal border
            border_x = min(col0, col1) * (self.cell_size + self.border_thickness)
            border_y = row0 * (self.cell_size + self.border_thickness)
            width = abs(col1 - col0) * (self.cell_size + self.border_thickness) + self.border_thickness
            return QtCore.QRectF(border_x, border_y, width, self.border_thickness)
        else:
            raise ValueError("Borders must be aligned either horizontally or vertically.")
