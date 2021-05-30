from abc import ABC, abstractmethod
from itertools import product


class Neighborhood(ABC):
    def __init__(self, distance):
        """Init neighborhood.
        Arguments:
            distance: max distance from the individual

        """
        self.distance = distance

    @abstractmethod
    def get_neighbours(self, grid_shape, idx):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid

        Return:
            Set of neighbours' positions.

        """
        pass


class LinearNeighborhood(Neighborhood):
    def get_neighbours(self, grid_shape, idx):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid

        Return:
            Set of neighbours' positions.

        """
        assert len(grid_shape) == len(idx)

        result = []

        # idx = (11, 24, 13) => position_0 = 11, position_1 = 24, etc.
        for axis, position in enumerate(idx):
            first_neighbor_pos = max(position - self.distance, 0)
            last_neighbor_pos = min(position + self.distance + 1, grid_shape[axis])

            for neighbor_position in range(first_neighbor_pos, last_neighbor_pos):
                neighbor_idx = list(idx)
                neighbor_idx[axis] = neighbor_position
                result.append(tuple(neighbor_idx))

        return result


class CompactNeighborhood(Neighborhood):
    def get_neighbours(self, grid_shape, idx):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid

        Return:
            Set of neighbours' positions.

        """
        assert len(grid_shape) == len(idx)

        ranges = []

        # idx = (11, 24, 13) => position_0 = 11, position_1 = 24, etc.
        for axis, position in enumerate(idx):
            first_neighbor_pos = max(position - self.distance, 0)
            last_neighbor_pos = min(position + self.distance + 1, grid_shape[axis])

            ranges.append(range(first_neighbor_pos, last_neighbor_pos))

        return list(product(*ranges))
