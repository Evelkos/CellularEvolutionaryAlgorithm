from abc import ABC, abstractmethod
from itertools import product


class Neighborhood(ABC):
    @abstractmethod
    def get_neighbours(self, grid_shape, idx, distance):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid
            distance: max distance from the individual

        Return:
            Set of neighbours' positions.

        """
        pass


class LinearNeighborhood(Neighborhood):
    @staticmethod
    def get_neighbours(grid_shape, idx, distance=1):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid
            distance: max distance from the individual

        Return:
            Set of neighbours' positions.

        """
        assert len(grid_shape) == len(idx)

        result = []

        # idx = (11, 24, 13) => position_0 = 11, position_1 = 24, etc.
        for axis, position in enumerate(idx):
            first_neighbor_pos = max(position - distance, 0)
            last_neighbor_pos = min(position + distance, grid_shape[axis])

            for neighbor_position in range(first_neighbor_pos, last_neighbor_pos + 1):
                neighbor_idx = list(idx)
                neighbor_idx[axis] = neighbor_position
                result.append(tuple(neighbor_idx))

        return result


class CompactNeighborhood(Neighborhood):
    @staticmethod
    def get_neighbours(grid_shape, idx, distance=1):
        """Get list of neighbours' positions on the grid.

        Arguments:
            grid_shape: shape of the grid that we want to get individuals from
            idx: individual's position on the grid
            distance: max distance from the individual

        Return:
            Set of neighbours' positions.

        """
        assert len(grid_shape) == len(idx)

        ranges = []

        # idx = (11, 24, 13) => position_0 = 11, position_1 = 24, etc.
        for axis, position in enumerate(idx):
            first_neighbor_pos = max(position - distance, 0)
            last_neighbor_pos = min(position + distance, grid_shape[axis])

            ranges.append(range(first_neighbor_pos, last_neighbor_pos + 1))

        return list(product(*ranges))
