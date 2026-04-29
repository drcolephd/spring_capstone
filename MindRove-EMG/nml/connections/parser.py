import os
import csv

class SequenceParser:
    """Handles loading and parsing a sequence configuration file."""

    def __init__(self, config_file: str, loop_delay: float = 0.010):
        """Initialize the parser with a configuration file."""
        self.config_file = config_file
        self.loop_sequence = []
        self.total_loop_time = 0
        self.loop_indices = []
        self.loop_delay = loop_delay  # Fixed delay for each loop iteration

    def load_config(self):
        """Load the loop sequence from the configuration file."""
        if not os.path.exists(self.config_file):
            print(f"Configuration file '{self.config_file}' not found. Falling back to default sequence.")
            return False

        self.loop_sequence = []
        self.total_loop_time = 0

        with open(self.config_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 4:
                    continue  # Skip malformed rows
                action_type, arg1, arg2, delay = row
                action_type = int(action_type)
                arg1 = arg1.strip()
                arg2 = arg2.strip()
                delay = float(delay)
                self.loop_sequence.append((action_type, arg1, arg2, delay))
                self.total_loop_time += delay

        self._generate_loop_indices()
        return True

    def _generate_loop_indices(self):
        """Generate the loop index mapping for each callback."""
        self.loop_indices = []
        cumulative_time = 0

        for action in self.loop_sequence:
            self.loop_indices.append(round(cumulative_time / self.loop_delay))
            cumulative_time += action[3]  # Add delay (4th column in the sequence)

    def get_sequence(self):
        """Return the parsed loop sequence, total loop time, and loop indices."""
        return self.loop_sequence, self.total_loop_time, self.loop_indices
