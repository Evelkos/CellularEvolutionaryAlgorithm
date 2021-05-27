from abc import ABC, abstractmethod


class Neighborhood(ABC):
    @abstractmethod
    def get_neighbours(self, grid, cell_loc):
        """Get list of cell's neighbours.

        Arguments:
            grid: grid that we want to get individuals from
            cell_loc: individual's position on the grid

        Return:
            List of cell's neighbours.

        """
        ...


class LinearNeighborhood(Neighborhood):
    def __init__(self, distance=1):
        """Init LinearNeighborhood.

        Arguments:
            distance: max distance from the cell
                distance = 1 => 5 individuals in the neighborhood
                distance = 2 => 9 individuals in the neighborhood

        """
        self.distance = distance


class CompactNeighborhood(Neighborhood):
    def __init__(self, distance=1):
        """Init CompactNeighborhood.

        Arguments:
            distance: max distance from the cell
                distance = 1 => 9 individuals in the neighborhood
                distance = 2 => 13 individuals in the neighborhood

        """
        self.distance = distance
