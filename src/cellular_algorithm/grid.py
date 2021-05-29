import numpy as np

from cellular_algorithm import Individual


class Grid:
    def __init__(self, shape, neighbourhood):
        """
        Arguments:
            shape: describes Grid's shape
            neighbourhood: type of neighbourhood

        """
        self.grid = np.empty(shape, dtype=Individual)
        self.neighbourhood = neighbourhood

    def __repr__(self):
        return f"{self.grid}, {self.neighbourhood}"

    @property
    def shape(self):
        return self.grid.shape

    def generate_individuals(self, boundaries):
        """Fill grid with random individuals.

        Split grid into discrits of equal size. Generate individual inside each discrit.

        Arguments:
            boundaries: todo
        """
        # Compute step for each grid dimension.
        steps = [
            (max(boundary) - min(boundary)) / individuals
            for individuals, boundary in zip(self.grid.shape, boundaries)
        ]
        # Get minimal possible values for individuals' coordinates.
        min_vals = [min(boundary) for boundary in boundaries]

        # For each position on the grid, compute coresponding discrit.
        for grid_position, _ in self.get_individuals():
            low = [
                idx * step + min_val
                for idx, step, min_val in zip(grid_position, steps, min_vals)
            ]
            high = [low_val + step for low_val, step in zip(low, steps)]

            individual = Individual(coordinates=np.random.uniform(low=low, high=high))
            self.set_individual(individual, grid_position)

    def set_individual(self, individual, grid_position):
        """Set new individual on the given position.

        Arguments:
            individual: individual we want to set on the given position
            grid_position: individual's position on the grid

        """
        self.grid[grid_position] = individual

    def get_individuals(self):
        for grid_position, individual in np.ndenumerate(self.grid):
            yield grid_position, individual
